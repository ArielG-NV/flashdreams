# Text-To-Video Demo Manifest Reference

`test-suite-configs/demo-manifest.yml` is a YAML manifest that defines prompt-only video generation fixtures and cases. Variants must exist in the global manifest and use one of the `text-to-video` demo's `supported-backends`.

## Fixture Keys

`prompt`: Text prompt passed to the runner.

`seed`: Deterministic generation seed.

`height`, `width`, `num_frames`: Optional target rollout shape hints consumed by demo adapters that expose those knobs.

## Fields

`fixtures`: Mapping from fixture id to prompt-only input data.

`cases`: Case entries with `id`, `fixture`, and `variant`.

`suites`: Suite slug to case ids.
