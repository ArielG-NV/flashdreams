---
title: 'Interactive Drive support'
---

<a id="apps-interactivedrive-interactivedrive-drivingsupport--interactive-drive-support"></a>

This package contains model-neutral scene loading, vehicle simulation,
conditioning rasterization, presentation, and input handling used by the
`interactive-drive` application.

The package owns its application arguments, default-scene download policy, and
`IApplication`/`ISession` implementations. Model-specific pipeline configs and
checkpoints are supplied by an adapter in `integrations_v2/<model>/`; this
package does not select a model.

Use the registered application entry points rather than importing an
integration directly:

```bash

uv run flashdreams-run-v2 interactive-drive-omnidreams --mode webrtc -- \
    --scene scene.usdz
```
