// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define main example_main
#include "../examples/launch.c"
#undef main

/* Run with SDL's dummy video driver; no model or CUDA work is performed. */
#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "Check failed at line %d: %s\n", __LINE__, #condition); \
    return 1; \
} } while (0)

static int poll_input(example_state *state, fd_input_event *event) {
    for (int tick = 0; tick < 10; tick++) {
        if (get_user_input_events(state, event) == 1) {
            return 1;
        }
        SDL_Delay(1);
    }
    return 0;
}

int main(void) {
    example_state state = {0};
    fd_input_event event = {0};
    CHECK(SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "dummy"));
    CHECK(open_window(&state, 640, 480, 30) == 0);
    SDL_FlushEvents(SDL_EVENT_FIRST, SDL_EVENT_LAST);
    CHECK(get_user_input_events(&state, &event) == 0);
    SDL_Delay(5100);
    CHECK(get_user_input_events(&state, &event) == 0);

    SDL_Event input = {0};
    input.type = SDL_EVENT_KEY_DOWN;
    input.key.key = SDLK_ESCAPE;
    input.key.down = true;
    CHECK(SDL_PushEvent(&input));
    CHECK(poll_input(&state, &event) == 1);
    CHECK(event.type == FD_KEY && strcmp(event.key, "Escape") == 0 && event.pressed);

    request_new_window_size(&state, 800, 600);
    int width, height;
    CHECK(SDL_GetWindowSize(state.window, &width, &height));
    CHECK(width == 800 && height == 600);
    SDL_FlushEvents(SDL_EVENT_FIRST, SDL_EVENT_LAST);
    input.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
    input.button.x = 400;
    input.button.y = 150;
    input.button.button = SDL_BUTTON_MIDDLE;
    input.button.down = true;
    CHECK(SDL_PushEvent(&input));
    CHECK(poll_input(&state, &event) == 1);
    CHECK(event.type == FD_MOUSE_BUTTON && event.button == 1 && event.pressed);
    CHECK(event.x == 0.5 && event.y == 0.25);

    input.type = SDL_EVENT_WINDOW_FOCUS_LOST;
    CHECK(SDL_PushEvent(&input));
    CHECK(poll_input(&state, &event) == 1);
    CHECK(event.type == FD_FOCUS && !event.pressed);
    CHECK(get_user_input_events(&state, &event) == 0);

    input.type = SDL_EVENT_QUIT;
    CHECK(SDL_PushEvent(&input));
    CHECK(poll_input(&state, &event) == 1 && event.type == FD_CLOSE);
    close_window(&state);
    puts("PASS: idle stays open, menu key, resized mouse coordinates, focus, manual close");
    return 0;
}
