import torch

import numpy as np
from projects.lingbot_world.camera_utils import SE3_inverse, compute_relative_poses


def compute_relative_poses_causal(
    c2ws_mat: torch.Tensor,
    trans_normalizer: float = 1.0,
    ref_pose: torch.Tensor | None = None,
) -> torch.Tensor:
    if ref_pose is None:
        ref_pose = c2ws_mat[0:1]
    assert ref_pose.shape == (1, 4, 4)
    c2ws_mat = torch.cat([ref_pose, c2ws_mat], dim=0)
    relative_poses = torch.bmm(
        SE3_inverse(c2ws_mat[:-1]), c2ws_mat[1:]
    )
    relative_poses[:, :3, 3] /= trans_normalizer
    return relative_poses


def test_compute_relative_poses_causal():
    camera_path = "assets/example_data/lingbot_world/poses.npy"
    poses = torch.from_numpy(np.load(camera_path)).float()

    relative_poses1, trans_normalizer = compute_relative_poses(poses, framewise=True)
    relative_poses2 = compute_relative_poses_causal(poses, trans_normalizer)
    torch.testing.assert_close(relative_poses1, relative_poses2, atol=1e-4, rtol=1e-4)
    
    last_pose = None
    relative_poses3 = []
    for pose in poses:
        pose = pose.unsqueeze(0)
        relative_pose = compute_relative_poses_causal(pose, trans_normalizer, last_pose)
        relative_poses3.append(relative_pose)
        last_pose = pose
    relative_poses3 = torch.cat(relative_poses3, dim=0)
    torch.testing.assert_close(relative_poses1, relative_poses3, atol=1e-4, rtol=1e-4)


# python -m projects.lingbot_world.tests.test_camera_utils
if __name__ == "__main__":
    test_compute_relative_poses_causal()