// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "flashdreams.h"
#include <SDL3/SDL.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct example_state {
    SDL_Window *window;
    char key[5];
    int frames;
    uint8_t *host_pixels;
    size_t host_capacity;
} example_state;

static int cuda_ok(CUresult status) {
    if (status != CUDA_SUCCESS) {
        const char *message = "Unknown CUDA error";
        cuGetErrorString(status, &message);
        fprintf(stderr, "CUDA: %s\n", message);
        return 0;
    }
    return 1;
}

static int open_window(void *context, int width, int height, int fps) {
    example_state *state = context;
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        fprintf(stderr, "SDL: %s\n", SDL_GetError());
        return -1;
    }
    state->window = SDL_CreateWindow("FlashDreams C window", width, height, SDL_WINDOW_RESIZABLE);
    if (!state->window) {
        fprintf(stderr, "SDL: %s\n", SDL_GetError());
        return -1;
    }
    printf("C window opened: %dx%d at %d fps\n", width, height, fps);
    return 0;
}

static int get_user_input_events(void *context, fd_input_event *event) {
    example_state *state = context;
    SDL_Event input;
    while (SDL_PollEvent(&input)) {
        if (input.type == SDL_EVENT_QUIT || input.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) {
            event->type = FD_CLOSE;
            return 1;
        }
        if (input.type == SDL_EVENT_KEY_DOWN || input.type == SDL_EVENT_KEY_UP) {
            if (input.key.repeat) {
                continue;
            }
            switch (input.key.key) {
                case SDLK_ESCAPE: event->key = "Escape"; break;
                case SDLK_RETURN: event->key = "Enter"; break;
                case SDLK_TAB: event->key = "Tab"; break;
                case SDLK_BACKSPACE: event->key = "Backspace"; break;
                case SDLK_UP: event->key = "ArrowUp"; break;
                case SDLK_DOWN: event->key = "ArrowDown"; break;
                case SDLK_LEFT: event->key = "ArrowLeft"; break;
                case SDLK_RIGHT: event->key = "ArrowRight"; break;
                case SDLK_LSHIFT: case SDLK_RSHIFT: event->key = "Shift"; break;
                case SDLK_LCTRL: case SDLK_RCTRL: event->key = "Control"; break;
                case SDLK_LALT: case SDLK_RALT: event->key = "Alt"; break;
                default:
                    if (input.key.key < 32 || input.key.key > 126) {
                        continue;
                    }
                    state->key[0] = (char)input.key.key;
                    state->key[1] = '\0';
                    event->key = state->key;
            }
            event->type = FD_KEY;
            event->pressed = input.key.down;
            return 1;
        }
        if (input.type == SDL_EVENT_MOUSE_MOTION ||
            input.type == SDL_EVENT_MOUSE_BUTTON_DOWN || input.type == SDL_EVENT_MOUSE_BUTTON_UP) {
            int width, height;
            if (!SDL_GetWindowSize(state->window, &width, &height) || width <= 0 || height <= 0) {
                continue;
            }
            int motion = input.type == SDL_EVENT_MOUSE_MOTION;
            event->type = motion ? FD_MOUSE_MOVE : FD_MOUSE_BUTTON;
            event->x = (motion ? input.motion.x : input.button.x) / width;
            event->y = (motion ? input.motion.y : input.button.y) / height;
            if (!motion) {
                if (input.button.button > SDL_BUTTON_RIGHT) {
                    continue;
                }
                event->button = input.button.button - 1;
                event->pressed = input.button.down;
            }
            return 1;
        }
        if (input.type == SDL_EVENT_MOUSE_WHEEL) {
            float direction = input.wheel.direction == SDL_MOUSEWHEEL_FLIPPED ? -1.0f : 1.0f;
            event->type = FD_MOUSE_WHEEL;
            event->x = input.wheel.x * direction;
            event->y = input.wheel.y * direction;
            return 1;
        }
        if (input.type == SDL_EVENT_WINDOW_FOCUS_GAINED || input.type == SDL_EVENT_WINDOW_FOCUS_LOST) {
            event->type = FD_FOCUS;
            event->pressed = input.type == SDL_EVENT_WINDOW_FOCUS_GAINED;
            return 1;
        }
    }
    return 0;
}

static int write_frames(void *context, const fd_cuda_frame *frame) {
    example_state *state = context;
    int width = frame->width, height = frame->height, frames = frame->frames;
    size_t frame_bytes = (size_t)width * (size_t)height * 3;
    if (width <= 0 || height <= 0 || frames <= 0 ||
        frame->byte_count != frame_bytes * (size_t)frames) {
        return -1;
    }
    CUdeviceptr source = (CUdeviceptr)(uintptr_t)frame->device_pointer;
    CUmemorytype memory_type;
    int device;
    if (!cuda_ok(cuPointerGetAttribute(&memory_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, source)) ||
        !cuda_ok(cuPointerGetAttribute(&device, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, source)) ||
        memory_type != CU_MEMORYTYPE_DEVICE || device != frame->device_index) {
        fprintf(stderr, "Expected a CUDA device buffer on the declared device.\n");
        return -1;
    }
    if (state->host_capacity < frame_bytes) {
        if (state->host_pixels && !cuda_ok(cuMemFreeHost(state->host_pixels))) {
            return -1;
        }
        state->host_pixels = NULL;
        state->host_capacity = 0;
        if (!cuda_ok(cuMemAllocHost((void **)&state->host_pixels, frame_bytes))) {
            return -1;
        }
        state->host_capacity = frame_bytes;
    }
    /* Read back one frame per batch for SDL's software window surface.
     * ponytail: CPU readback limits throughput; use GPU interop for a fast presenter.
     */
    if (!cuda_ok(cuMemcpyDtoHAsync(state->host_pixels, source, frame_bytes, (CUstream)frame->stream)) ||
        !cuda_ok(cuStreamSynchronize((CUstream)frame->stream))) {
        return -1;
    }
    SDL_Surface *source_surface = SDL_CreateSurfaceFrom(
        width, height, SDL_PIXELFORMAT_RGB24, state->host_pixels, width * 3);
    SDL_Surface *destination = SDL_GetWindowSurface(state->window);
    int presented = source_surface && destination &&
        SDL_BlitSurfaceScaled(source_surface, NULL, destination, NULL, SDL_SCALEMODE_LINEAR) &&
        SDL_UpdateWindowSurface(state->window);
    SDL_DestroySurface(source_surface);
    if (!presented) {
        fprintf(stderr, "SDL: %s\n", SDL_GetError());
        return -1;
    }
    state->frames += frames;
    return 0;
}

static void request_new_window_size(void *context, int width, int height) {
    example_state *state = context;
    if (!SDL_SetWindowSize(state->window, width, height)) {
        fprintf(stderr, "SDL: %s\n", SDL_GetError());
    }
}

static void close_window(void *context) {
    example_state *state = context;
    SDL_DestroyWindow(state->window);
    state->window = NULL;
    SDL_Quit();
    if (state->host_pixels) {
        cuda_ok(cuMemFreeHost(state->host_pixels));
        state->host_pixels = NULL;
        state->host_capacity = 0;
    }
    printf("C window closed after %d frames\n", state->frames);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s /absolute/path/to/python [--no-c-window] [app args...]\n", argv[0]);
        return 1;
    }
    example_state state = {0};
    fd_client_window window = {0};
    window.struct_size = sizeof(window);
    window.abi_version = FD_ABI_VERSION;
    window.user_data = &state;
    window.open = open_window;
    window.get_user_input_events = get_user_input_events;
    window.write = write_frames;
    window.close = close_window;
    window.request_new_window_size = request_new_window_size;

    const char *defaults[] = {
        "crazy-robotaxi-omnidreams", "--mode", "c", "--",
        "--display-width", "1280", "--display-height", "720"
    };
    int use_defaults = argc == 2;
    int offset = 2;
    const fd_client_window *selected_window = &window;
    if (argc > 2 && strcmp(argv[2], "--no-c-window") == 0) {
        selected_window = NULL;
        offset++;
    }
    char error[2048];
    int result = fd_run(argv[1],
                        use_defaults ? (int)(sizeof(defaults) / sizeof(defaults[0])) : argc - offset,
                        use_defaults ? defaults : (const char *const *)(argv + offset),
                        selected_window, error, sizeof(error));
    if (result) {
        fprintf(stderr, "FlashDreams launch failed: %s\n", error);
        return result;
    }
    puts("FlashDreams returned control to C successfully.");
    return 0;
}
