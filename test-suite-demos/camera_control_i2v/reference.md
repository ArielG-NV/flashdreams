# Camera-Control I2V Demo Manifest Reference

`test-suite-configs/demo-manifest.yml` is a YAML manifest that defines prompt + first-frame + camera/action control fixtures. Variants must exist in the global manifest and use one of the `camera-control-i2v` demo's `supported-backends`.

## Fixture Keys

`first-frame`: Asset id or environment variable resolved by the demo adapter.

`prompt`: Text prompt paired with the first frame.

`camera_path`: Compact camera command timeline interpreted by the demo adapter.

`actions`: Optional model-specific action timeline for backends that support action controls.

`seed`, `chunk_size`, `num_steps`: Optional rollout controls.

## Fields

`fixtures`: Mapping from fixture id to camera-control input data.

`cases`: Case entries with `id`, `fixture`, and `variant`.

`suites`: Suite slug to case ids.
