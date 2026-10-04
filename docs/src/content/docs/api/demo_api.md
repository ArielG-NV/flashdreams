---
title: 'Experimental demo API'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

`flashdreams.runtime.demo` is the shared, higher-level demo API layered over
the [inference API](inference_api.md). It owns scenarios, I/O modes,
drivers, warmup, replay, benchmarking, metrics, and presentation. See the
[API overview](index.md) to compare all API families.

## Run a replay

An integration supplies the adapter, supported modes, and scenario schema.

```python

from flashdreams.runtime.demo import (
    DemoAdapter,
    DemoSpec,
    NullOutputSpec,
    run_replay_demo,
)

def replay(adapter: DemoAdapter, scenario):
    return run_replay_demo(
        spec=DemoSpec(
            model_id=adapter.model_id,
            input_mode="replay",
            output=NullOutputSpec(),
            scenario=scenario,
        ),
        adapter=adapter,
    )

```

## Output modes

Choose `NullOutputSpec`, `Mp4OutputSpec`,
`LocalWindowOutputSpec`, `IOFactoryOutputSpec`, or
`WebRTCOutputSpec`. The adapter must support the chosen mode. Finite
replay and benchmark helpers reject WebRTC output.

## Benchmarking, warmup, and metrics

`run_benchmark_demo` runs one finite scenario and writes benchmark stats.
Adapter-provided warmup sessions use the same runtime but are excluded from
measured statistics. Set `capture_output=False` when output is unnecessary.

## API reference

All names below are importable directly from `flashdreams.runtime.demo`.

### Specifications and adapter boundary

<a id="flashdreams.runtime.demo.DemoAdapter"></a>
### `DemoAdapter`

```text
class DemoAdapter ( * args , ** kwargs )
```

Bases: `ModelAdapter`, `Protocol`

Transport-neutral model adapter consumed by demo runners.

<a id="flashdreams.runtime.demo.DemoAdapter.supported_input_modes"></a>
#### `supported_input_modes`

```text
supported_input_modes ( ) → tuple [ str , ... ]
```

Return demo input modes this adapter can prepare.
<a id="flashdreams.runtime.demo.DemoAdapter.supported_output_modes"></a>
#### `supported_output_modes`

```text
supported_output_modes ( ) → tuple [ str , ... ]
```

Return demo output modes this adapter can run.
<a id="flashdreams.runtime.demo.DemoAdapter.prepare_scenario"></a>
#### `prepare_scenario`

```text
prepare_scenario ( spec : DemoSpec ) → PreparedScenario
```

Validate and materialize scenario inputs before runtime creation.

<a id="flashdreams.runtime.demo.DemoSpec"></a>
### `DemoSpec`

```text
class DemoSpec ( * , model_id: str , input_mode: str , output: ~flashdreams.runtime.demo.spec.NullOutputSpec | ~flashdreams.runtime.demo.spec.Mp4OutputSpec | ~flashdreams.runtime.demo.spec.LocalWindowOutputSpec | ~flashdreams.runtime.demo.spec.IOFactoryOutputSpec | ~flashdreams.runtime.demo.spec.WebRTCOutputSpec , preset_id: str | None = None , scenario: ~typing.Any | None = None , config: ~flashdreams.runtime.config.InferenceConfig | None = None , metadata: ~collections.abc.Mapping[str , ~typing.Any] = <factory> )
```

Bases: `object`

User-facing shared demo run description.

<a id="flashdreams.runtime.demo.PreparedScenario"></a>
### `PreparedScenario`

```text
class PreparedScenario ( * , initial_inputs: ~flashdreams.runtime.inputs.InferenceInput , user_inputs: ~flashdreams.runtime.inputs.UserInputs = <factory> , source_schema: ~flashdreams.runtime.inputs.UserInputSchema = <factory> , canonicalizer: ~flashdreams.runtime.canonical.InputCanonicalizer = <factory> , mapping: ~flashdreams.runtime.mapping.InputMapping | None = None , metadata: ~collections.abc.Mapping[str , ~typing.Any] = <factory> )
```

Bases: `object`

Runtime-ready scenario prepared by a model demo adapter.

<a id="flashdreams.runtime.demo.NullOutputSpec"></a>
### `NullOutputSpec`

```text
class NullOutputSpec ( * , mode : Literal [ 'null' ] = 'null' , store_results : bool = False )
```

Bases: `object`

Headless/null replay output.

<a id="flashdreams.runtime.demo.Mp4OutputSpec"></a>
### `Mp4OutputSpec`

```text
class Mp4OutputSpec ( * , path : str | Path , fps : int | float | None = None , mode : Literal [ 'mp4' ] = 'mp4' , output_layout : Literal [ 'tchw' , 'btchw' , 'bcthw' , 'bvtchw' ] | None = 'bvtchw' , move_to_cpu : bool = True )
```

Bases: `object`

MP4 replay output.

<a id="flashdreams.runtime.demo.LocalWindowOutputSpec"></a>
### `LocalWindowOutputSpec`

```text
class LocalWindowOutputSpec ( * , mode : Literal [ 'local-window' ] = 'local-window' , fps : float | None = None , title : str = 'FlashDreams' )
```

Bases: `object`

Native local-window presentation output.

<a id="flashdreams.runtime.demo.LocalWindowOutputSpec.fps"></a>
#### `fps`

```text
fps : float | None
```

Presentation rate; `None` uses application session metadata.
<a id="flashdreams.runtime.demo.LocalWindowOutputSpec.title"></a>
#### `title`

```text
title : str
```

Native window title.

<a id="flashdreams.runtime.demo.IOFactoryOutputSpec"></a>
### `IOFactoryOutputSpec`

```text
class IOFactoryOutputSpec ( * , mode : Literal [ 'io-factory' ] = 'io-factory' )
```

Bases: `object`

Opaque output owned by an application `IOFactory`.

<a id="flashdreams.runtime.demo.WebRTCOutputSpec"></a>
### `WebRTCOutputSpec`

```text
class WebRTCOutputSpec ( * , mode : Literal [ 'webrtc' ] = 'webrtc' , host : str = '127.0.0.1' , port : int = 8080 , fps : int = 30 , video_width : int = 1280 , video_height : int = 720 , warmup_chunks : int = 0 , warmup_timeout_s : float = 30.0 , client_liveness_timeout_s : float = 30.0 , web_dir : str | Path | None = None , request_session_path : str = '/request_session' , preload_name : str | None = None )
```

Bases: `object`

Shared WebRTC serving output.

### Convenience runners and results

<a id="flashdreams.runtime.demo.run_replay_demo"></a>
### `run_replay_demo`

```text
run_replay_demo ( *, spec: ~flashdreams.runtime.demo.spec.DemoSpec, adapter: ~flashdreams.runtime.demo.spec.DemoAdapter, output_target_factory: ~collections.abc.Callable[[~flashdreams.runtime.demo.spec.NullOutputSpec | ~flashdreams.runtime.demo.spec.Mp4OutputSpec | ~flashdreams.runtime.demo.spec.LocalWindowOutputSpec | ~flashdreams.runtime.demo.spec.IOFactoryOutputSpec | ~flashdreams.runtime.demo.spec.WebRTCOutputSpec], ~flashdreams.runtime.output.OutputTarget] | None = None, output_sink_factory: ~collections.abc.Callable[[~flashdreams.runtime.demo.spec.NullOutputSpec | ~flashdreams.runtime.demo.spec.Mp4OutputSpec | ~flashdreams.runtime.demo.spec.LocalWindowOutputSpec | ~flashdreams.runtime.demo.spec.IOFactoryOutputSpec | ~flashdreams.runtime.demo.spec.WebRTCOutputSpec], ~flashdreams.demo.io.OutputSink] = <function build_output_sink>, metrics: ~flashdreams.runtime.metrics.MetricsRecorder | None = None ) → RunResult
```

Run one prepared replay scenario through the shared batch demo path.

<a id="flashdreams.runtime.demo.run_benchmark_demo"></a>
### `run_benchmark_demo`

```text
run_benchmark_demo ( * , spec : DemoSpec , adapter : DemoAdapter , stats_path : str | Path | None = None , stats_dir : str | Path | None = None , capture_output : bool = True , metrics : MetricsRecorder | None = None , mp4_writer : Callable [ [ ... ] , Path ] | None = None , pipeline : StepPipeline | None = None ) → RunResult
```

Run one benchmarked demo session through `BenchmarkRunMode`.

<a id="flashdreams.runtime.demo.RunResult"></a>
### `RunResult`

```text
class RunResult ( * , status : Literal [ 'completed' , 'failed' , 'skipped' , 'cancelled' , 'rejected' , 'not_activated' ] , artifacts : Sequence [ OutputArtifact ] = () , metrics : MetricsSnapshot | None = None , reason : str | None = None , error : Exception | None = None )
```

Bases: `object`

Outcome of one demo session.

<a id="flashdreams.runtime.demo.RunResult.rejected"></a>
#### `rejected`

```text
classmethod rejected ( reason : str ) → RunResult
```

Admission refused the session. The only no-session result helper.

### Extension points

<a id="flashdreams.runtime.demo.RuntimeHost"></a>
### `RuntimeHost`

```text
class RuntimeHost ( runtime : InferenceRuntime , * , worker : ModelExecutionWorker | None = None , is_control_rank : bool = True , worker_loop : Callable [ [ ] , None ] | None = None )
```

Bases: `object`

Own one runtime and the worker used for model-affine calls.

<a id="flashdreams.runtime.demo.RuntimeHost.runtime"></a>
#### `runtime`

```text
property runtime : InferenceRuntime
```

Return the hosted runtime.
<a id="flashdreams.runtime.demo.RuntimeHost.worker"></a>
#### `worker`

```text
property worker : ModelExecutionWorker
```

Return the host’s model-execution worker.
<a id="flashdreams.runtime.demo.RuntimeHost.is_control_rank"></a>
#### `is_control_rank`

```text
property is_control_rank : bool
```

Whether this process owns run modes, providers, sinks, and metrics.
<a id="flashdreams.runtime.demo.RuntimeHost.is_healthy"></a>
#### `is_healthy`

```text
property is_healthy : bool
```

Return whether admission should continue accepting sessions.
<a id="flashdreams.runtime.demo.RuntimeHost.unhealthy_reason"></a>
#### `unhealthy_reason`

```text
property unhealthy_reason : str | None
```

Return the first latched unhealthy reason, if any.
<a id="flashdreams.runtime.demo.RuntimeHost.unhealthy_error"></a>
#### `unhealthy_error`

```text
property unhealthy_error : Exception | None
```

Return the first latched unhealthy error, if any.
<a id="flashdreams.runtime.demo.RuntimeHost.mark_unhealthy"></a>
#### `mark_unhealthy`

```text
mark_unhealthy ( reason : str = 'marked unhealthy' , error : Exception | None = None ) → None
```

Latch the host as unhealthy without overwriting the first reason.
<a id="flashdreams.runtime.demo.RuntimeHost.preload"></a>
#### `preload`

```text
preload ( ) → None
```

Initialize optional distributed state and preload runtime resources.
<a id="flashdreams.runtime.demo.RuntimeHost.warmup"></a>
#### `warmup`

```text
warmup ( plan : ModelWarmupPlan | None = None ) → None
```

Run warmup sessions through the same worker boundary as real sessions.
<a id="flashdreams.runtime.demo.RuntimeHost.call"></a>
#### `call`

```text
call ( func : Callable [ [ ... ] , _T ] , / , * args : object , ** kwargs : object ) → _T
```

Run one model-affine callable synchronously on the worker.
<a id="flashdreams.runtime.demo.RuntimeHost.call_async"></a>
#### `call_async`

```text
async call_async ( func : Callable [ [ ... ] , _T ] , / , * args : object , ** kwargs : object ) → _T
```

Run model-affine work without blocking realtime event loops.
<a id="flashdreams.runtime.demo.RuntimeHost.start_session"></a>
#### `start_session`

```text
start_session ( inputs : InferenceInput ) → InferenceSession
```

Start one inference session through the hosted runtime.
<a id="flashdreams.runtime.demo.RuntimeHost.run_worker_loop"></a>
#### `run_worker_loop`

```text
run_worker_loop ( ) → None
```

Serve control-rank work on non-control ranks until runtime shutdown.
<a id="flashdreams.runtime.demo.RuntimeHost.close"></a>
#### `close`

```text
close ( ) → None
```

Close runtime-owned state and stop the model-execution worker.

<a id="flashdreams.runtime.demo.RunMode"></a>
### `RunMode`

```text
class RunMode ( * args , ** kwargs )
```

Bases: `Protocol`

Run/session construction strategy consumed by shared helpers.

<a id="flashdreams.runtime.demo.RunContext"></a>
### `RunContext`

```text
class RunContext ( host: RuntimeHost , run_metrics: SessionMetricsRecorder , admission: AdmissionPolicy , model_warmup_plan: ModelWarmupPlan = <factory> , services: Mapping[str , object] = <factory> , cleanup_tasks: set[asyncio.Task[RunResult]] = <factory> )
```

Bases: `object`

Run-scoped services shared by one or more demo sessions.

<a id="flashdreams.runtime.demo.SessionEdges"></a>
### `SessionEdges`

```text
class SessionEdges ( input_source: InputSource, output_sink: OutputSink, cleanup_tasks: set[asyncio.Task[RunResult]], metrics: SessionMetricsRecorder = <factory>, error_policy: ErrorPolicy = <factory>, transport: TransportService = <factory>, clock: RealtimeClock | DeterministicClock | None = None, activation: ActivationPolicy | None = None )
```

Bases: `object`

Per-session input/output/policy bundle consumed by drivers.

<a id="flashdreams.runtime.demo.SessionEdges.is_closed"></a>
#### `is_closed`

```text
property is_closed : bool
```

Return whether `close_result(...)` has already finalized this session.
<a id="flashdreams.runtime.demo.SessionEdges.record_cleanup_error"></a>
#### `record_cleanup_error`

```text
record_cleanup_error ( exc : Exception ) → None
```

Record a cleanup error without letting metrics failures block teardown.
<a id="flashdreams.runtime.demo.SessionEdges.record_orphaned_cleanup"></a>
#### `record_orphaned_cleanup`

```text
record_orphaned_cleanup ( exc : Exception ) → None
```

Record timed-out worker cleanup without blocking teardown.
<a id="flashdreams.runtime.demo.SessionEdges.close_result"></a>
#### `close_result`

```text
close_result ( * , status : Literal [ 'completed' , 'failed' , 'skipped' , 'cancelled' , 'not_activated' ] = 'completed' , reason : str | None = None , error : Exception | None = None ) → RunResult
```

Idempotently close output, transport, and metrics once.

<a id="flashdreams.runtime.demo.run_demo_session"></a>
### `run_demo_session`

```text
run_demo_session ( * , context : RunContext , spec : DemoSpec , scenario : PreparedScenario , adapter : DemoAdapter , run_mode : RunMode , pipeline : StepPipeline , reservation : SessionReservation | None = None ) → RunResult
```

Run one prepared demo session through a selected run mode.

<a id="flashdreams.runtime.demo.run_demo_session_async"></a>
### `run_demo_session_async`

```text
async run_demo_session_async ( * , context : RunContext , spec : DemoSpec , scenario : PreparedScenario , adapter : DemoAdapter , run_mode : RunMode , pipeline : StepPipeline , reservation : SessionReservation | None = None ) → RunResult
```

Run one prepared async/realtime demo session through a selected run mode.
