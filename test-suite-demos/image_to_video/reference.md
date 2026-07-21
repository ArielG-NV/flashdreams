# Image-To-Video Demo Manifest Reference

`test-suite-configs/demo-manifest.yml` is a YAML manifest that defines prompt + first-frame generation fixtures and cases. Variants must exist in the global manifest and use one of the `image-to-video` demo's `supported-backends`.

## Fixture Keys

`first-frame`: Asset id or environment variable resolved by the demo adapter.

`prompt`: Text prompt paired with the first frame.

`seed`, `height`, `width`, `num_frames`: Optional rollout controls when the backend exposes them.

## Fields

`fixtures`: Mapping from fixture id to image-to-video input data.

`cases`: Case entries with `id`, `fixture`, and `variant`.

`suites`: Suite slug to case ids.
