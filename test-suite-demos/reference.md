# Global Test-Suite Manifest Reference

`public-manifest.yml` is the top-level YAML catalog for describing model compatability with various demos, primarily for testing purposes.

This file lists: model backends, decoders, variants (backend+decoder+... configurations), demos, and test suites.

## Add A Backend, Decoder, Or Demo

Add a backend under the yaml `backends` when a model package exists under `integrations/<model>`. Keep checkpoint environment-variable checks inside that integration package; do not add checkpoint-specific fields to this manifest.

Add a decoder under the yaml `decoders` when a decoder adapter exists under `decoders/<decoder>`. `variants` can specify the decoder a model is supposed to use via specifying the `settings.decoder` yaml key.

Add a demo under the yaml `demos` with an adapter path, the path to that demo's `test-suite-configs/demo-manifest.yml`, and `supported-backends` based on the `backends` yaml key. To add a new `demo` add the demo adapter to `test-suite-demos/`.

## Run Test Suites

```bash
## Run all
# runs all defined test `cases`, in each demo defined
uv run test-suite test-suite-demos/public-manifest.yml --all
# runs all defined test `cases`, in the interactive-drive demo
uv run test-suite test-suite-demos/public-manifest.yml --all --demo interactive-drive

## Run by backend
# runs all defined test `cases` using the cosmos-predict2 backend, in each demo defined
uv run test-suite test-suite-demos/public-manifest.yml --backend cosmos-predict2
# runs all defined test `cases` using the cosmos-predict2 backend, in the interactive-drive demo
uv run test-suite test-suite-demos/public-manifest.yml --backend cosmos-predict2 --demo interactive-drive

## Run by suite
# runs all defined test `cases` using the per_commit suite, in each demo defined
uv run test-suite test-suite-demos/public-manifest.yml --suite per_commit
# runs all defined test `cases` using the per_commit suite, in the interactive-drive demo
uv run test-suite test-suite-demos/public-manifest.yml --suite per_commit --demo interactive-drive

## Run by variant
# runs all defined test `cases` using the cosmos-predict2-wan21-baseline variant, in each demo defined
uv run test-suite test-suite-demos/public-manifest.yml --variant cosmos-predict2-wan21-baseline
# runs all defined test `cases` using the cosmos-predict2-wan21-baseline variant, in the interactive-drive demo
uv run test-suite test-suite-demos/public-manifest.yml --variant cosmos-predict2-wan21-baseline --demo interactive-drive

## Run a configuration evaluation pass
# runs all possible permutations of configurations for a `<backend_id>`, allowing for evaluation of the best configuration for that backend.
uv run test-suite test-suite-demos/public-manifest.yml --backend-configuration-test cosmos-predict2
# runs all possible permutations of configurations for a `<backend_id>`, allowing for evaluation of the best configuration for that backend for a particular demo.
uv run test-suite test-suite-demos/public-manifest.yml --backend-configuration-test cosmos-predict2 --demo interactive-drive
```

When explicitly naming variants, results write to `artifacts/test-suite-results/<MANIFEST_NAME>/<demo>/<variant>/cases/<case_id>/`.
- Each case directory contains the model output plus `result.yml` and `aux-test-info.yml`. The latter packages the test input, fixture, variant, command, and captured standard output/error.

When running a `--backend-configuration-test <backend_id>` for a backend to evaluate configurations, results write the configurations to `artifacts/test-suite-results/<MANIFEST_NAME>/run-all-<backend_id>/configuration-list.yml` and results to `.../<demo>/<configuration_variant_id>/`.
- CURRENTLY UNIMPLEMENTED: A tool that identifies the "best" configuration from the results for a particular goal, such as the fastest-generation or highest quality.

## Run Eval Suites

```bash
uv run eval-suite --target-variant-dir artifacts/test-suite-results/public-manifest/interactive-drive/cosmos-predict2-wan21-baseline --output artifacts/eval/cosmos-predict2-wan21-baseline

uv run eval-suite --reference-variant-dir artifacts/test-suite-results/public-manifest/interactive-drive/cosmos-predict2-wan21-baseline   --target-variant-dir artifacts/test-suite-results/public-manifest/interactive-drive/cosmos-predict2-wan21-optimized   --output artifacts/eval/cosmos-predict2-wan21-optimized
```

`eval.json` is for target-only checks such as blank screen, grey frame, blur, and stripe failures. `relative-eval.json` is for reference-relative checks such as MAE, RMSE, PSNR, and FLIP.

## Bundled Demo Catalog

`text-to-video`: Prompt-only video generation. Compatible backends currently include Cosmos-Predict2, Wan 2.1, Self-Forcing, Causal-Forcing, and FastVideo CausalWan 2.2 variants.

`image-to-video`: Prompt + first-frame video generation. Compatible backends currently include Cosmos-Predict2 I2V, Wan 2.1 I2V, Wan 2.2 TI2V, and Causal-Forcing I2V variants.

`interactive-drive`: Keyboard-input + first-frame + prompt interactive driving/world-model cases. Compatible backends currently include Omnidreams-style interactive variants and related control-capable world backends.

`camera-control-i2v`: Prompt + first-frame + camera/action controls. Compatible backends currently include LingBot World and HY-WorldPlay variants.

`video-super-resolution`: Input-video enhancement cases. Compatible backend currently includes FlashVSR.

## Legacy Video-Quality Regression

Video-quality regression cases currently remain in the standalone `configs/video_quality_cases.yml` catalog. That file is legacy test collateral and should be planned for deprecation separately from the new global backend/demo manifest schema. Run the calibration suite with:

```bash
uv run python -m flashdreams.quality.video_quality.run_regression --manifest configs/video_quality_cases.yml --suite calibration
```

`eval.json` and `relative-eval.json` remain output artifact names; they are not schema manifests.

## Field Reference

`schema_version`: Integer schema version. The current version is `1`.

`backends`: List of model backend entries. Each entry has `id` and `adapter`. The adapter is a dotted import path to an integration-owned backend adapter.

`decoders`: List of decoder entries. Each entry has `id` and `adapter`. The adapter is a dotted import path to a decoder-owned adapter.

`named-variants`: List of reusable variant entries. Each entry has `id`, `backend`, and `settings`. `backend` must refer to a declared backend id. `settings` is adapter-owned data; when it includes `decoder` or `decoders`, those values must refer to declared decoder ids.

`demos`: List of demo entries. Each entry has `id`, `adapter`, `config-path`, and `supported-backends`. `config-path` points to the demo's per-demo manifest. `supported-backends` lists backend ids that can satisfy the demo input contract, such as prompt-only T2V, prompt + first-frame I2V, keyboard-input interactive drive, camera/action-control I2V, or video super-resolution.

`suites`: Mapping of suite slug to demo ids. Every demo id in every suite must exist in `demos`.

