# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cheap source-level guards for the optional Vulkan/CUDA interop path."""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.ci_cpu

_PACKAGE = Path(__file__).parents[1] / "ludus_renderer"


def _between(source: str, start: str, end: str) -> str:
    return source.split(start, 1)[1].split(end, 1)[0]


def test_vulkan_hot_path_uses_reused_zero_copy_interop() -> None:
    renderer = (_PACKAGE / "_cpp/render/ludus_timestamped_vk.cpp").read_text()
    vkutil = (_PACKAGE / "_cpp/common/vkutil.cpp").read_text()
    binding = (_PACKAGE / "_cpp/bindings/torch_rasterize_vk.cpp").read_text()
    interactive = (
        Path(__file__).parents[2] / "omnidreams/interactive_drive/rasterizer.py"
    ).read_text()
    render = _between(
        renderer, "void ludusRenderBatchVk", "void ludusCopyBatchResultsVk"
    )
    torch_render = _between(
        binding,
        "torch::Tensor ludus_timestamped_render_batch_vk",
        "std::tuple<int, bool> ludus_timestamped_render_to_staging_vk",
    )

    # Fixed-function color/depth ordering resolves visibility before one
    # compute pass writes the exported, device-local linear allocation.
    for shader in _PACKAGE.glob("shaders/*.frag.glsl"):
        source = shader.read_text()
        assert "layout(early_fragment_tests) in;" in source
        assert "layout(location = 0) out vec4 out_color;" in source
        assert "binding = 14" not in source
        assert "output_pixels.rgba8[pixel_index]" not in source
    export = (_PACKAGE / "shaders/ts_export.comp.glsl").read_text()
    assert "binding = 14" in export
    assert "binding = 15" in export
    assert "imageLoad(color_image" in export
    assert "output_pixels.rgba8[pixel_index]" in export
    assert "subpass.colorAttachmentCount = 1" in renderer
    assert (
        "VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT" in renderer
    )
    assert "VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT" in render
    assert "VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT" in render
    assert "vkCmdDispatch" in render
    assert "resizeExternalBuffer(s.vkctx, outputBuffer, totalSize" in render
    assert "VK_BUFFER_USAGE_STORAGE_BUFFER_BIT" in render
    assert "cuExternalMemoryGetMappedBuffer" in vkutil
    assert "cuExternalMemoryGetMappedMipmappedArray" not in vkutil
    assert "VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT" in vkutil
    assert "vkCmdCopyImageToBuffer" not in renderer
    assert "torch::from_blob" in torch_render
    assert "ludusMapBatchResultsVk" in torch_render
    assert "ludusCopyBatchResultsVk" not in torch_render
    assert '"vulkan": LudusVulkanTimestampedContext' in interactive

    # Output and color/depth allocations are capacity/cache reused; teardown
    # synchronization remains behind those rare growth paths.
    assert "target.maxLayers >= layers" in renderer
    assert "externalBufferNeedsResize(outputBuffer, totalSize)" in render
    assert "LUDUS_VK_OUTPUT_SLOTS" in binding
    assert "cudaStreamSynchronize" not in render
    assert "vkDeviceWaitIdle" not in render


def test_vulkan_growth_preserves_existing_scenes_and_required_features() -> None:
    renderer = (_PACKAGE / "_cpp/render/ludus_timestamped_vk.cpp").read_text()
    vkutil = (_PACKAGE / "_cpp/common/vkutil.cpp").read_text()

    assert "growCapacityGeometrically" in renderer
    assert "finishSharedBufferGrowth" in renderer
    assert "growth.preserveBytes" in renderer
    assert "cudaMemcpyAsync" in _between(
        renderer,
        "static void finishSharedBufferGrowth",
        "static void rebuildRenderPassAndPipelines",
    )
    assert "features2.features.geometryShader = VK_TRUE" in vkutil
    # Fragment shaders no longer perform storage writes, so this feature is not
    # needed by the corrected graphics-plus-compute path.
    assert "features2.features.fragmentStoresAndAtomics = VK_TRUE" not in vkutil


def test_vulkan_handoff_is_timeline_and_ownership_ordered() -> None:
    renderer = (_PACKAGE / "_cpp/render/ludus_timestamped_vk.cpp").read_text()
    vkutil = (_PACKAGE / "_cpp/common/vkutil.cpp").read_text()
    vkheader = (_PACKAGE / "_cpp/common/vkutil.h").read_text()
    plugin = (_PACKAGE / "_ops/_plugin_vk.py").read_text()
    render = _between(
        renderer, "void ludusRenderBatchVk", "void ludusCopyBatchResultsVk"
    )
    mapping = _between(
        renderer, "uint8_t* ludusMapBatchResultsVk", "void ludusCopyBatchResultsVk"
    )

    # Interop selects native handles at compile time. OS handles only transport
    # the Vulkan allocation/semaphore into CUDA; renderer state tracks the CUDA
    # import itself, so there is no Linux-fd field or Windows handle leak.
    assert "VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME" in vkutil
    assert "VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME" in vkutil
    assert "CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32" in vkutil
    assert "CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD" in vkutil
    assert "CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32" in vkutil
    assert "CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD" in vkutil
    assert "CloseHandle(semaphoreHandle)" in vkutil
    assert "CloseHandle(memoryHandle)" in vkutil
    assert "memFd" not in vkheader
    assert "VK_USE_PLATFORM_WIN32_KHR" in vkheader
    assert 'ldflags = ["cuda.lib", "vulkan-1.lib", "nvjpeg.lib"]' in plugin
    assert "Vulkan backend is currently only supported on Linux" not in plugin
    assert "cuSignalExternalSemaphoresAsync" in vkutil
    assert "cuWaitExternalSemaphoresAsync" in vkutil
    assert "signalInteropTimelineFromCuda" in render
    assert "waitInteropTimelineOnCuda" in mapping
    assert "VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO" in render
    assert "VK_QUEUE_FAMILY_EXTERNAL" in render
    assert "srcQueueFamilyIndex = s.vkctx.graphicsQueueFamily" in render
    assert "dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL" in render
    assert "VK_ACCESS_SHADER_WRITE_BIT" in render
    assert "VK_ACCESS_TRANSFER_WRITE_BIT" not in render


def test_vulkan_push_constant_updates_cover_the_declared_stage_range() -> None:
    renderer = (_PACKAGE / "_cpp/render/ludus_timestamped_vk.cpp").read_text()
    stage_mask = _between(
        renderer,
        "static constexpr VkShaderStageFlags kLudusPushConstantStages =",
        ";",
    )

    assert "VK_SHADER_STAGE_TASK_BIT_EXT" in stage_mask
    assert "VK_SHADER_STAGE_MESH_BIT_EXT" in stage_mask
    assert "VK_SHADER_STAGE_FRAGMENT_BIT" in stage_mask
    assert "VK_SHADER_STAGE_COMPUTE_BIT" in stage_mask
    assert "pushRange.stageFlags = kLudusPushConstantStages" in renderer

    push_calls = re.findall(r"vkCmdPushConstants\((.*?)\);", renderer, re.DOTALL)
    assert push_calls
    assert all("kLudusPushConstantStages" in call for call in push_calls)
