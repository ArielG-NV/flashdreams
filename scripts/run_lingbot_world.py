import os
import argparse

import torch
import numpy as np
import cv2
import mediapy as media
from einops import rearrange
from huggingface_hub import login as huggingface_login

from flashsim.distributed import init as distributed_init
from flashsim.configs.lingbot_world import LINGBOT_WORLD_CONFIGS
from flashsim.io.s3_sync import sync_s3_dir_to_local
from flashsim.pipeline.lingbot_world import ProfileEvents


def SE3_inverse(T: torch.Tensor) -> torch.Tensor:
    Rot = T[:, :3, :3]  # [B,3,3]
    trans = T[:, :3, 3:]  # [B,3,1]
    R_inv = Rot.transpose(-1, -2)
    t_inv = -torch.bmm(R_inv, trans)
    T_inv = torch.eye(4, device=T.device, dtype=T.dtype)[None, :, :].repeat(
        T.shape[0], 1, 1
    )
    T_inv[:, :3, :3] = R_inv
    T_inv[:, :3, 3:] = t_inv
    return T_inv


def compute_relative_poses(
    c2ws_mat: torch.Tensor,
    framewise: bool = False,
    normalize_trans: bool = True,
) -> torch.Tensor:
    ref_w2cs = SE3_inverse(c2ws_mat[0:1])
    relative_poses = torch.matmul(ref_w2cs, c2ws_mat)
    # ensure identity matrix for 1st frame
    relative_poses[0] = torch.eye(4, device=c2ws_mat.device, dtype=c2ws_mat.dtype)
    if framewise:
        # compute pose between i and i+1
        relative_poses_framewise = torch.bmm(
            SE3_inverse(relative_poses[:-1]), relative_poses[1:]
        )
        relative_poses[1:] = relative_poses_framewise
    if normalize_trans:  # note refer to camctrl2: "we scale the coordinate inputs to roughly 1 standard deviation to simplify model learning."
        translations = relative_poses[:, :3, 3]  # [f, 3]
        max_norm = torch.norm(translations, dim=-1).max()
        # only normlaize when moving
        if max_norm > 0:
            relative_poses[:, :3, 3] = translations / max_norm
    return relative_poses


def create_meshgrid(
    n_frames: int,
    height: int,
    width: int,
    bias: float = 0.5,
    device="cuda",
    dtype=torch.float32,
) -> torch.Tensor:
    x_range = torch.arange(width, device=device, dtype=dtype)
    y_range = torch.arange(height, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).view([-1, 2]) + bias  # [h*w, 2]
    grid_xy = grid_xy[None, ...].repeat(n_frames, 1, 1)  # [f, h*w, 2]
    return grid_xy


def get_plucker_embeddings(
    c2ws_mat: torch.Tensor,
    Ks: torch.Tensor,
    height: int,
    width: int,
    only_rays_d: bool = False,
):
    n_frames = c2ws_mat.shape[0]
    grid_xy = create_meshgrid(
        n_frames, height, width, device=c2ws_mat.device, dtype=c2ws_mat.dtype
    )  # [f, h*w, 2]
    fx, fy, cx, cy = Ks.chunk(4, dim=-1)  # [f, 1]

    i = grid_xy[..., 0]  # [f, h*w]
    j = grid_xy[..., 1]  # [f, h*w]
    zs = torch.ones_like(i)  # [f, h*w]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs

    directions = torch.stack([xs, ys, zs], dim=-1)  # [f, h*w, 3]
    directions = directions / directions.norm(dim=-1, keepdim=True)  # [f, h*w, 3]

    rays_d = directions @ c2ws_mat[:, :3, :3].transpose(-1, -2)  # [f, h*w, 3]
    if only_rays_d:
        plucker_embeddings = rays_d  # [f, h*w, 3]
        plucker_embeddings = plucker_embeddings.view(
            [n_frames, height, width, 3]
        )  # [f*h*w, 3]
    else:
        rays_o = c2ws_mat[:, :3, 3]  # [f, 3]
        rays_o = rays_o[:, None, :].expand_as(rays_d)  # [f, h*w, 3]
        # rays_dxo = torch.cross(rays_o, rays_d, dim=-1) # [f, h*w, 3]
        # note refer to: apt2
        plucker_embeddings = torch.cat([rays_o, rays_d], dim=-1)  # [f, h*w, 6]
        plucker_embeddings = plucker_embeddings.view(
            [n_frames, height, width, 6]
        )  # [f*h*w, 6]
    return plucker_embeddings


parser = argparse.ArgumentParser()
parser.add_argument(
    "--total_blocks", type=int, default=60, help="Total blocks to generate."
)
parser.add_argument(
    "--overwrite_config_name", type=str, default=None, help="Overwrite config name."
)
parser.add_argument("--video_height", type=int, default=464, help="Video height.")
parser.add_argument("--video_width", type=int, default=832, help="Video width.")
args = parser.parse_args()

EXAMPLE_DATA_DIR_S3 = "s3://flashsim/assets/example_data/lingbot_world"
EXAMPLE_DATA_DIR_LOCAL = os.path.join(
    os.path.dirname(__file__), "../assets/example_data/lingbot_world"
)

CAMERA_NAMES = ["default"]
DATA = [
    {
        "pose_path": os.path.join(EXAMPLE_DATA_DIR_LOCAL, "poses.npy"),
        "intrinsic_path": os.path.join(EXAMPLE_DATA_DIR_LOCAL, "intrinsics.npy"),
        "first_frame_path": os.path.join(EXAMPLE_DATA_DIR_LOCAL, "image.jpg"),
        "text_prompt_path": os.path.join(EXAMPLE_DATA_DIR_LOCAL, "prompt.txt"),
    }
    for _ in CAMERA_NAMES
]
CONFIG_NAME = "LingBot-World-Fast"

if args.overwrite_config_name is not None:
    CONFIG_NAME = args.overwrite_config_name
print(f"Running Lingbot World inference with config: {CONFIG_NAME}")

# download example data from S3
CREDENTIAL_PATH = os.path.join(
    os.path.dirname(__file__), "../credentials/s3_checkpoint.secret"
)
assert os.path.exists(CREDENTIAL_PATH), (
    f"Credential file not found at {CREDENTIAL_PATH}"
)
sync_s3_dir_to_local(
    s3_dir=EXAMPLE_DATA_DIR_S3,
    s3_credential_path=CREDENTIAL_PATH,
    cache_dir=EXAMPLE_DATA_DIR_LOCAL,
    max_workers=10,
    show_progress=True,
    verify_checksum=True,
    desc="Syncing from S3",
)

# login huggingface
HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN is not None, "HF_TOKEN is not set"
huggingface_login(HF_TOKEN)
print("logged in to huggingface")

# initialize distributed inference
distributed_init()
world_size = torch.distributed.get_world_size()
rank = torch.distributed.get_rank()
print(f"initialized distributed training with world size {world_size} and rank {rank}")
device = torch.device(f"cuda:{rank}")
dtype = torch.bfloat16

# prepare data
plucker_videos = []
first_frames = []
prompts = []
for data in DATA:
    first_frame = media.read_image(data["first_frame_path"])
    first_frame = cv2.resize(first_frame, (args.video_width, args.video_height))
    first_frame = (
        torch.from_numpy(first_frame).to(dtype=dtype, device=device) / 127.5 - 1.0
    )  # range [-1, 1]
    first_frame = rearrange(first_frame, "h w c -> 1 c h w")  # [1, C, H, W]
    first_frames.append(first_frame)

    Ks = np.load(data["intrinsic_path"])  # [N, 4]
    Ks = torch.from_numpy(Ks).to(device=device, dtype=torch.float32)
    c2ws = np.load(data["pose_path"])  # [N, 4, 4]
    c2ws = torch.from_numpy(c2ws).to(device=device, dtype=torch.float32)
    c2ws = compute_relative_poses(c2ws, framewise=True)
    plucker_video = get_plucker_embeddings(
        c2ws, Ks, args.video_height, args.video_width
    )
    plucker_video = rearrange(plucker_video, "t h w c -> t c h w")  # [T, C, H, W]
    plucker_videos.append(plucker_video.to(dtype=dtype))

    prompt = open(data["text_prompt_path"], "r").readlines()[0]
    prompts.append(prompt)
first_frames = torch.stack(first_frames, dim=0).unsqueeze(0)  # [B, V, 1, C, H, W]
plucker_videos = torch.stack(plucker_videos, dim=0).unsqueeze(0)  # [B, V, T, C, H, W]
prompts = [prompts]  # [B, V]
batch_size, num_views, plucker_num_frames, _3, height, width = plucker_videos.shape
print("loaded plucker_videos.shape:", plucker_videos.shape)

# initialize pipeline
pipeline_config = LINGBOT_WORLD_CONFIGS[CONFIG_NAME]
pipeline_config.seed += rank
pipeline = pipeline_config.setup(device=device)
cache = pipeline.initialize_cache(text=prompts, image=first_frames)

torch.cuda.synchronize()
if torch.distributed.is_initialized():
    torch.distributed.barrier()

# streaming inference
start = 0
generated_video = []
for i in range(args.total_blocks):
    num_frames = pipeline.get_num_frames(i)
    end = start + num_frames
    if end > plucker_num_frames:
        break
    print(
        f"autoregressive_index: {i}, num_frames: {num_frames}, start: {start}, end: {end}"
    )
    generated_video.append(
        pipeline.streaming_inference(
            autoregressive_index=i,
            plucker=plucker_videos[:, :, start:end],
            cache=cache,
        )
    )
    start = end
    pipeline.finalize(
        autoregressive_index=i,
        cache=cache,
    )  # update KV cache for the next block
generated_video = torch.cat(generated_video, dim=2)  # [B, V, T, C, H, W], range [-1, 1]
generated_num_frames = generated_video.shape[2]
print("end of streaming inference, generated_video.shape:", generated_video.shape)

if rank == 0:
    # print profiling results.
    torch.cuda.synchronize()
    ProfileEvents.finalize(cache.profile_events, skip_first_n=3)

    # export result
    canvas = rearrange(generated_video, "1 v t c h w -> t h (v w) c")
    canvas = (canvas.float().cpu().numpy() + 1.0) / 2.0  # range [0, 1]
    canvas = (canvas * 255).astype(np.uint8)
    save_path = f"outputs/{CONFIG_NAME}_{world_size}gpus.mp4"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    media.write_video(save_path, canvas, fps=16)
    print(f"saved generated video to {save_path}")

if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()
