# Interactive-Drive Demo Manifest Reference

`test-suite-configs/demo-manifest.yml` defines reusable complete driving scenes for `interactive-drive`. Cases are launched through the `interactive-drive` application, not `flashdreams-run`: the application reads HD map, camera calibration, initial images, and prompts from the ClipGT USDZ archive.

## Add A Fixture

Every fixture must provide `clipgt`. It accepts either a local ClipGT USDZ archive:

```yaml
clipgt: /datasets/clipgt-0d404ff7-2b66-498c-b047-1ed8cded60d4.usdz
```

or a direct Hugging Face file URL. The test suite caches this download through `huggingface_hub` before launch:

```yaml
clipgt: https://huggingface.co/datasets/nvidia/omni-dreams-scenes/resolve/main/scenes/clipgt-0d404ff7-2b66-498c-b047-1ed8cded60d4.usdz
```

`hf://datasets/nvidia/omni-dreams-scenes/scenes/clipgt-<uuid>.usdz?revision=main` is also accepted. The archive must be a `clipgt-*.usdz` file, and Hugging Face access needs `HF_TOKEN` when the repository requires it.

Optional fixture keys are `prompt`, `perf-manifest`, `scene-variant`, `hf-org`, `stream-mjpeg`, and `stop-after-chunks`. The default test execution streams to `127.0.0.1:18080` and exits after one consumed chunk; this makes it suitable for a non-desktop GPU host.

## Add A Case

Add a case with `id`, `fixture`, and `variant`. The fixture must exist in this file. Its variant must exist globally and use one of this demo's `supported-backends`.

## Field Reference

`schema_version`: Integer schema version. The current version is `1`.

`fixtures`: Mapping from fixture id to complete ClipGT scene input and optional launch settings. Each executed case records its selected drive view as `generated.mp4` in the case output directory.

`cases`: List of case entries. Each case has `id`, `fixture`, and `variant`.

`suites`: Mapping of suite slug to case ids. Every case id in every suite must exist in `cases`.
