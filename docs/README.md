# FlashDreams documentation

This directory hosts the Sphinx sources for the FlashDreams API
reference site. The layout follows the
[`gsplat`](https://github.com/nerfstudio-project/gsplat/tree/main/docs)
project.

## Build locally

The doc build only needs the packages listed in `requirements.txt`, but
`autodoc` imports `flashdreams` itself, so the build environment must
have the project installed (the workspace `uv sync` already does this).

```bash
# from the repo root
uv pip install -r docs/requirements.txt
uv run --package flashdreams make -C docs html
```

The rendered site lands in `docs/_build/html/index.html`. Open it with
any browser, e.g. `xdg-open docs/_build/html/index.html`.

## Layout

```
docs/
├── Makefile                # standard Sphinx build dispatch
├── requirements.txt        # doc-only Python deps
└── source/
    ├── conf.py             # Sphinx configuration (theme + extensions)
    ├── index.rst           # landing page + top-level toctree
    ├── apis/
    │   ├── core.rst        # flashdreams.core (attention, distributed, …)
    │   ├── infra.rst       # flashdreams.infra (pipeline, diffusion, …)
    │   ├── recipes.rst     # flashdreams.recipes (alpadreams, wan, …)
    │   └── serving.rst     # placeholder for the future serving layer
    └── examples/           # one rst per inference launcher
        ├── alpadreams.rst
        ├── self_forcing.rst
        ├── causal_forcing.rst
        ├── causal_wan22.rst
        ├── lingbot_world.rst
        └── wan21.rst
```

## Hosting on GitHub Pages

`.github/workflows/doc.yml` builds the docs on every push / PR /
release and pushes the rendered HTML to the `gh-pages` branch
(layout cribbed from
[`gsplat`](https://github.com/nerfstudio-project/gsplat/blob/main/.github/workflows/doc.yml)):

| Trigger                | Deployed under                  | Banner shows |
| ---------------------- | ------------------------------- | ------------ |
| `push` to `main`       | `gh-pages:/main/`               | `main`       |
| `release` (tag)        | `gh-pages:/versions/<ver>/`     | `<ver>`      |
| `pull_request`         | (build only, no deploy)         | n/a          |
| `workflow_dispatch`    | `gh-pages:/versions/<ver>/`     | `<ver>`      |

One-time GitHub setup after the first run:

1. **Settings → Pages** → set *Source* to **Deploy from a branch**,
   branch = `gh-pages`, folder = `/ (root)`.
2. (Optional) point a custom domain at it and uncomment the
   `cname:` line in `doc.yml`.
3. Each release also appends its version to
   `gh-pages:/versions/index.txt`, useful for a future version-picker
   widget on the site.

## Adding new content

- **A new model recipe** — append a section to `source/apis/recipes.rst`
  using `.. automodule:: flashdreams.recipes.<name>`, and add a launcher
  walk-through to `source/examples/<name>.rst`. Wire the new file into
  the matching toctree in `source/index.rst` (autoregressive vs
  bidirectional vs serving).
- **A new infra component** — re-export the public symbols from the
  package `__init__.py`, then add an `.. autoclass::` block to the
  relevant section of `source/apis/infra.rst`.
- **A new API category** — drop a new `source/apis/<topic>.rst`, add it
  to `index.rst`, and (optionally) introduce a new captioned toctree.
