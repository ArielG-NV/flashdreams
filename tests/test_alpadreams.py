import torch
from flashsim.configs.alpadreams import AlpadreamsPipelineConfig


def test_alpadreams_streaming_inference():
    num_views = 1
    height = 704
    width = 1280

    device = torch.device("cuda")
    dtype = torch.bfloat16

    image = torch.randn(1, num_views, 1, 3, height, width, device=device, dtype=dtype)
    text = [["Hello, world!"] * num_views]

    pipeline = AlpadreamsPipelineConfig().setup(device=device)
    cache = pipeline.initialize_cache(text=text, image=image)

    autoregressive_index = 0
    num_frames = pipeline.get_num_frames(autoregressive_index)
    hdmap = torch.randn(
        1, num_views, num_frames, 3, height, width, device=device, dtype=dtype
    )
    decoded_video = pipeline.streaming_inference(
        autoregressive_index, hdmap=hdmap, cache=cache
    )
    pipeline.finalize(autoregressive_index, cache=cache)
    assert decoded_video.shape == hdmap.shape

    autoregressive_index = 1
    num_frames = pipeline.get_num_frames(autoregressive_index)
    hdmap = torch.randn(
        1, num_views, num_frames, 3, height, width, device=device, dtype=dtype
    )
    decoded_video = pipeline.streaming_inference(
        autoregressive_index, hdmap=hdmap, cache=cache
    )
    pipeline.finalize(autoregressive_index, cache=cache)
    assert decoded_video.shape == hdmap.shape


if __name__ == "__main__":
    test_alpadreams_streaming_inference()
