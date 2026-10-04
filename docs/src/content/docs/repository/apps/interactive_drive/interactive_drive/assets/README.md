---
title: 'Assets'
---

<a id="apps-interactivedrive-interactivedrive-assets-readme--assets"></a>

This directory holds scene-loading compatibility helpers
(`scene_bundle.py`) plus the bundled HUD control sprites under
`wheel_and_pedals/`.

<a id="apps-interactivedrive-interactivedrive-assets-readme--wheelandpedals"></a>

## `wheel_and_pedals/`

Steering-wheel and pedal PNGs used by the `InteractiveDriveUILoop`
Dear ImGui HUD:

- `steering_wheel.png`
- `throttle_pressed.png`, `throttle_unpressed.png`
- `brake_pressed.png`, `brake_unpressed.png`

The application loads these files from the installed package when it creates
the HUD. They are included as package data by
`apps/interactive_drive/pyproject.toml`.

<a id="apps-interactivedrive-interactivedrive-assets-readme--scenes"></a>

## Scenes

Scene USDZs are not stored here. By default, the application downloads its
scene from the `nvidia/omni-dreams-scenes` Hugging Face dataset; pass
`--scene PATH` as an application argument to use a local USDZ instead.
