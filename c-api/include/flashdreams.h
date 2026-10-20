// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef FLASHDREAMS_H
#define FLASHDREAMS_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#ifdef FLASHDREAMS_CAPI_BUILD
#define FD_API __declspec(dllexport)
#else
#define FD_API __declspec(dllimport)
#endif
#else
#define FD_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define FD_ABI_VERSION 2

typedef enum fd_event_type {
    FD_CLOSE = 1,
    FD_RESET = 2,
    FD_KEY = 3,
    FD_MOUSE_MOVE = 4,
    FD_MOUSE_BUTTON = 5,
    FD_MOUSE_WHEEL = 6,
    FD_FOCUS = 7
} fd_event_type;

typedef struct fd_input_event {
    int32_t type;
    /* Microseconds since open; zero asks the runtime to timestamp receipt. */
    uint64_t timestamp_us;
    /* UTF-8 KeyboardEvent.key, valid until the next get_user_input_events call. */
    const char *key;
    /* Normalized mouse coordinates (0..1), or wheel deltas for FD_MOUSE_WHEEL.
     * Mouse buttons: 0 = left, 1 = middle, 2 = right. */
    double x;
    double y;
    int32_t button;
    /* 0/1: key/button released/pressed, or viewport unfocused/focused. */
    int32_t pressed;
} fd_input_event;

/* Packed CUDA uint8 RGB24, [frames][height][width][3]. The descriptor is
 * borrowed for the write callback only. device_pointer must not be dereferenced
 * by the CPU. device_index identifies the CUDA device/primary context.
 *
 * stream is a borrowed CUstream/cudaStream_t on device_index (NULL is a valid
 * default stream). Pixel production is ordered before work submitted to it.
 * Enqueue all reads/copies on this stream before returning from write; the
 * allocation remains alive until that work completes. A different consumer
 * stream must wait for production and join back onto this stream before return.
 * Do not destroy the stream, free the pointer, retain the descriptor, or submit
 * further work using the pointer after write returns.
 */
typedef struct fd_cuda_frame {
    const void *device_pointer;
    size_t byte_count;
    int width;
    int height;
    int frames;
    int64_t step_index;
    int device_index;
    void *stream;
} fd_cuda_frame;

/* All callbacks use the C calling convention and run on the calling/UI thread.
 * Keep this table, user_data, and callback code alive until fd_run returns.
 * Callbacks must not throw or reenter fd_run.
 */
typedef struct fd_client_window {
    size_t struct_size;
    uint32_t abi_version;
    void *user_data;
    /* Required. Return 0 on success; nonzero aborts the run. */
    int (*open)(void *user_data, int width, int height, int fps);
    /* Required. Return 1 and fill event, 0 if drained, or -1 on error.
     * Consume each event once. The runtime drains at most 4096 per UI tick.
     */
    int (*get_user_input_events)(void *user_data, fd_input_event *event);
    /* Required. CUDA buffer and stream as described above. The callback runs
     * with device_index current. Return 0 on success; nonzero aborts the run.
     */
    int (*write)(void *user_data, const fd_cuda_frame *frame);
    /* Required. Must tolerate repeated calls and partially failed open. */
    void (*close)(void *user_data);

    /* Optional IClientWindow requests. */
    void (*request_hide_cursor)(void *user_data, int hide);
    void (*request_lock_cursor_to_window)(void *user_data, int lock);
    void (*request_new_window_size)(void *user_data, int width, int height);
} fd_client_window;

/* Blocking launch of a registered v2 app. argv has the same arguments as
 * flashdreams-run-v2 (without the executable name, including -- for app args).
 * A non-NULL window supplies callbacks for --mode c; other modes still work.
 * --mode c enables embedded cleanup and requires the callback table above.
 * NULL uses normal modes and defaults. Supply the absolute Python executable
 * of the environment
 * containing FlashDreams when initializing Python; NULL uses CPython defaults.
 * If Python is already initialized, python_executable is ignored.
 *
 * Returns 0 on success/help, 1 on failure. Writes a truncated, NUL-terminated
 * error into error_buffer when non-NULL and error_capacity > 0.
 * Call serially from one host thread; Python remains initialized between runs.
 * The host must not finalize Python or unload this library during a run.
 */
FD_API int fd_run(const char *python_executable, int argc,
                  const char *const *argv, const fd_client_window *window,
                  char *error_buffer, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
