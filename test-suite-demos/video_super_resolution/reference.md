# Video Super-Resolution Demo Manifest Reference

`test-suite-configs/demo-manifest.yml` is a YAML manifest that defines input-video enhancement fixtures and cases. Variants must exist in the global manifest and use one of the `video-super-resolution` demo's `supported-backends`.

## Fixture Keys

`input-video`: Asset id or environment variable resolved by the demo adapter.

`scale`: Requested upsampling scale when the backend exposes it.

`seed`: Deterministic generation seed for stochastic stages.

## Fields

`fixtures`: Mapping from fixture id to VSR input data.

`cases`: Case entries with `id`, `fixture`, and `variant`.

`suites`: Suite slug to case ids.
