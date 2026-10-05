---
title: 'Contributing to FlashDreams'
---

<a id="contributing--contributing-to-flashdreams"></a>

Thanks for your interest in contributing to **FlashDreams**. This project
is developed openly on GitHub and released under the
[Apache License 2.0](https://github.com/NVIDIA/flashdreams/blob/main/LICENSE). Outside contributions — bug reports,
feature requests, performance improvements, new model integrations,
documentation fixes — are genuinely welcome, and this guide explains how
they fit in alongside the project's day-to-day work.

We have intentionally kept the process light. If something below is
unclear or feels heavier than it should be, that's a bug; please file an
issue and we'll fix it.

<a id="contributing--table-of-contents"></a>

## Table of contents

1. [Ways to contribute](CONTRIBUTING.md#contributing--ways-to-contribute)
2. [Project governance](CONTRIBUTING.md#contributing--project-governance)
3. [Developer Certificate of Origin (DCO)](CONTRIBUTING.md#contributing--developer-certificate-of-origin-dco)
4. [Submitting a pull request](CONTRIBUTING.md#contributing--submitting-a-pull-request)
5. [Code review and merge](CONTRIBUTING.md#contributing--code-review-and-merge)
6. [File Tree Of FlashDreams](CONTRIBUTING.md#contributing--file-tree-of-flashdreams)
7. [Adding documentation](CONTRIBUTING.md#contributing--adding-documentation)
8. [Adding an Agent Skill](CONTRIBUTING.md#contributing--adding-an-agent-skill)
9. [Coding conventions](CONTRIBUTING.md#contributing--coding-conventions)
10. [Testing](CONTRIBUTING.md#contributing--testing)
11. [Dependency version bounds](CONTRIBUTING.md#contributing--dependency-version-bounds)
12. [Working with a single integration package](CONTRIBUTING.md#contributing--working-with-a-single-integration-package)
13. [Versioning and publishing](CONTRIBUTING.md#contributing--versioning-and-publishing)
14. [Licensing of contributions](CONTRIBUTING.md#contributing--licensing-of-contributions)
15. [Reporting issues](CONTRIBUTING.md#contributing--reporting-issues)
16. [Code of Conduct](CONTRIBUTING.md#contributing--code-of-conduct)

<a id="contributing--ways-to-contribute"></a>

## Ways to contribute

There are several useful ways to help out, ordered roughly from "low
overhead" to "high overhead":

- **Try FlashDreams and tell us what broke.** A clear bug report — what
  you ran, what you expected, what you saw — is one of the most valuable
  contributions a project of this kind can receive.
- **Improve documentation.** README clarifications, integration walkthroughs,
  performance notes, and FAQ entries all land easily and benefit every
  future user.
- **Fix bugs.** Issues labelled `good first issue` are a friendly
  starting point. Larger fixes are welcome too — please leave a comment
  on the relevant issue first so we can avoid duplicate work.
- **Add or extend integrations.** New video-generation models, new schedulers,
  new integrations. For non-trivial features, please open a design issue
  before sending the PR (see [Submitting a pull request](CONTRIBUTING.md#contributing--submitting-a-pull-request)).
- **Performance work.** FlashDreams cares about latency and throughput
  on NVIDIA GPUs. Numbers and reproducible benchmarks make these PRs
  easy to evaluate.

<a id="contributing--project-governance"></a>

## Project governance

FlashDreams was developed inside NVIDIA's Simulation & Imitation
Learning group, and at the time of release NVIDIA holds the
maintainer and admin roles on the
[NVIDIA/flashdreams](https://github.com/NVIDIA/flashdreams) repository.
That includes the `main` branch protections, release tags, the package
publishing keys, and the right to merge.

We treat that as a starting point, not an endpoint. Our intent — modelled
on projects like [Slang](https://github.com/shader-slang/slang), which
moved from a single-vendor home to community governance once it grew an
external user base — is to open governance up as a contributor community
develops. Concretely, that means:

- **Review responsibility follows subsystem ownership.** As contributors
  take long-term ownership of an area, they can become required reviewers
  for changes there, NVIDIA employee or not.
- **Decisions happen in public.** Significant design changes are
  discussed in GitHub issues, pull requests, or
  [Discussions](https://github.com/NVIDIA/flashdreams/discussions).
  Internal NVIDIA roadmap planning that touches the public project will
  surface as a public issue before it lands.
- **Release notes credit external contributors** by name and PR.
- **Open path to maintainer.** Contributors who consistently land
  high-quality work in an area, participate in reviews, and engage with
  the issue tracker can be invited to become maintainers. There is no
  fixed time bar; sustained good judgment is what we look for.

If you have feedback on governance — including things you'd like to see
formalised faster — please open a Discussion. We'd rather hear it than
not.

<a id="contributing--developer-certificate-of-origin-dco"></a>

## Developer Certificate of Origin (DCO)

**This project will only accept contributions under the Apache-2.0
license.** By submitting a pull request you agree that your
contribution is licensed under the Apache License, Version 2.0 (see
[LICENSE](https://github.com/NVIDIA/flashdreams/blob/main/LICENSE)).

All contributions to FlashDreams are made under the
[Developer Certificate of Origin](https://developercertificate.org/).
This is a lightweight, well-understood mechanism (used by the Linux
kernel, GitLab, NVIDIA TensorRT, and many other projects) that lets you
attest that you have the right to submit your contribution under the
project's license — without requiring a separate Contributor License
Agreement.

Full text of the [Developer Certificate of Origin](https://developercertificate.org/):

```text

Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.

```

Pull requests without DCO sign-off will be asked to rebase before merge.
This is a hard gate; please don't take a polite ping personally.

NVIDIA contributors and external contributors follow the *same* DCO
process. NVIDIA-internal IP review, where applicable, is handled by
NVIDIA reviewers on your behalf — you do not need to engage with it as
an outside contributor.

<a id="contributing--signing-your-work"></a>

### Signing Your Work

We require that all contributors sign-off on their commits. This
certifies that the contribution is your original work, or you have
rights to submit it under the same license, or a compatible license.
Any contribution which contains commits that are not Signed-Off will
not be accepted.

To sign off on a commit, use the `--signoff` (or `-s`) option when
committing your changes:

```bash

$ git commit -s -m "Add cool feature."

```

This will append the following trailer to your commit message:

```text

Signed-off-by: Your Name <your@email.com>

```

The `user.name` and `user.email` git config values must be set to your
real name and a verifiable email address — sign-offs from anonymous or
pseudonymous identities cannot be accepted.

<a id="contributing--submitting-a-pull-request"></a>

## Submitting a pull request

The short version:

1. Fork the repo on GitHub and create a feature branch from `main`.
2. Make your changes. Keep PRs small and focused; a 200-line PR that
   does one thing reviews much faster than a 2000-line PR that does ten.
3. Add or update tests where it makes sense. Every test must carry
   a CI tier marker (`@pytest.mark.ci_cpu`, `@pytest.mark.ci_gpu`, or
   `@pytest.mark.manual`); see
   [Testing](CONTRIBUTING.md#contributing--testing) below. The project enforces this at collection
   time -- pytest will error if a test is missing a marker or combines
   `ci_cpu` with `ci_gpu`. A per-test `manual` marker may override a
   module-level CI marker.
4. Run the project's checks locally:

```bash

```

   uv run --group lint pre-commit run -a       # format + lint + type-check
   uv run --group test pytest -m ci_cpu         # CPU tests (no GPU required)

5. Sign off your commits (`git commit --signoff`) and push to your fork.
6. Open a pull request against `main`. Include any context a reviewer
   would need that isn't obvious from the diff, and link the issue your
   PR resolves if there is one.

For larger features (a new integration or a substantial refactor), please
open an issue first to discuss the design. This
saves everyone time and gives you a chance to surface trade-offs before
investing implementation effort.

<a id="contributing--code-review-and-merge"></a>

## Code review and merge

- Every pull request requires CI to pass and at least one approving
  review from a maintainer responsible for the touched area.
- For changes that span multiple subsystems, expect reviews from each
  affected area's maintainers.
- We squash-merge to keep `main`'s history readable. The PR title and
  description become the squash commit message — please make them
  descriptive and reviewer-facing.
- `main` is gated by GitHub's **merge queue**. Once your PR is approved
  and CI is green, click "Merge when ready" — GitHub will rebase your
  branch on top of `main` plus any earlier queued PRs, re-run the
  required checks against that combined state, and only land the merge
  if everything is still green. You do not need to manually rebase or
  re-run CI when another PR lands first. PRs that would conflict or
  fail after rebase are kicked back out of the queue automatically.
  Maintainers: do not enable "Require branches to be up to date before
  merging" alongside the queue — they're redundant, and enabling both
  reintroduces the rebase-storm problem the queue exists to solve.
- If a review comment is unclear, ask. We'd rather have a 30-second
  clarifying exchange than a misunderstanding turning into rework.

We aim for an initial review on every PR within two business days. If
your PR has been quiet longer than that, please feel free to leave a
short ping comment.

<a id="contributing--file-tree-of-flashdreams"></a>

## File Tree Of FlashDreams

Where a new app or model goes. Each documents its own layout:

```text

apps/                           # reusable apps; layout in apps/README.md
integrations_v2/                # model packages; layout in integrations_v2/README.md

```

The framework package, and the test and doc trees:

```text

flashdreams/flashdreams/        # the framework package
  core/                         # numerical primitives, checkpoint loading, attention, I/O
  infra/                        # framework contracts: configs, pipelines, encoders/decoders, schedulers, runners
  recipes/                      # built-in reusable recipe code (WAN, Cosmos, TAEHV, ...)
  api_v2/                       # separate application protocols implemented by v2 apps
  runtime_v2/                   # the two-thread loop that runs api_v2 applications
  runtime/                      # experimental model-inference API
    demo/                       # higher-level demo API: modes, drivers, warmup, replay, benchmarks
  serving/                      # optional serving utilities (WebRTC, network, launch)
  demo/                         # shared transport-neutral application hosting and I/O primitives
  accelerated/                  # accelerated kernels (quantization, multi-head attention)
  quality/                      # output-quality regression utilities (video, CLIP compare)
  configs/                      # runner registry and CLI aggregator
  plugins/                      # external-runner plugin layer (RunnerConfig discovery)
  scripts/                      # console-script entry points (flashdreams-run)
  _pytest_plugins/              # pytest plugins (e.g. CI-tier marker enforcement)

flashdreams/test_v2/            # FlashDreams Runtime/Protocol tests (window, run_session, threads)
flashdreams/tests/              # framework tests not yet migrated to test_v2/
tests/                          # repo-wide test-runner scripts + meta checks, not package tests
docs/src/content/docs/           # Zensical Markdown pages

```

See the [API overview](https://github.com/NVIDIA/flashdreams/blob/main/docs/src/content/docs/api/index.md)
before choosing an integration boundary.

<a id="contributing--adding-documentation"></a>

## Adding documentation

Canonical documentation lives under `docs/src/content/docs/**` and uses Markdown.
Add new canonical pages as `.md` files. Markdown outside that directory must be a one-line
link to the relevant page and section, with
exactly two exceptions:

- `skills/**/SKILL.md` contains the actual instructions for a repository
  Agent Skill and must stay under `skills/`.
- `flashdreams/README.md` contains its own concise package introduction. It
  may link to canonical pages, but it must not be reduced to a link-only stub.

Prefer a subsection in an existing topic page. Add a file only when the topic
is a distinct technical reference. Keep general guidance short, direct, and in
plain English; detailed developer material may use precise technical language.
Link to the exact existing section rather than copying it, and delete references
to removed files or commands. Zensical builds the sidebar from the pages under
`docs/src/content/docs/`. API documentation must name
the correct surface: `flashdreams.runtime` is the inference API,
`flashdreams.runtime.demo` is the higher-level demo API, and
`flashdreams.api_v2` is the separate application protocol API. Start from the
[API overview](../api/index.md).

Every page under `docs/src/content/docs/repository/integrations_v2/**` uses the standard
**Integration links** section with **Applications** and **Configuration**
links.

<a id="contributing--build-and-preview-the-documentation"></a>

### Build and preview the documentation

Run these commands from the repository root:

```bash

python tools/check_docs_layout.py
uv run --only-group docs zensical build -f docs/zensical.toml
uv run --only-group docs zensical serve -f docs/zensical.toml
```

The first command checks the repository's documentation rules. The build
command renders the site once; `serve` starts a live preview at
`http://localhost:8000`.
The docs workflow builds the same Zensical site on pull requests without
publishing them. Updates from `main` and releases are published to GitHub
Pages.

<a id="contributing--adding-an-agent-skill"></a>

## Adding an Agent Skill

Create a skill at `skills/<lowercase-hyphenated-name>/SKILL.md`. The file must
have YAML frontmatter containing a matching `name` and a specific
`description` that says what the skill does and when to use it. Keep common
policy in this contributing guide or canonical documentation and link to it;
do not copy the skill into `docs/src/content/docs/`.

Add the new skill to the **Skill Map** in `AGENTS.md` so agents can discover
it. Keep `SKILL.md` focused; add sibling references, scripts, or assets only
when they materially improve the workflow. Validate skill structure and the
documentation layout with:

```bash

uv run --group test pytest -m ci_cpu tests/test_agent_skills.py tests/test_docs_layout.py

```

### Using repository skills locally

Skills are opt-in. Link all of them into the directory used by your agent, or
link only the skills you want:

```bash

mkdir -p .agents
ln -s ../skills .agents/skills

```

Replace `.agents` with `.cursor` or `.claude` for those clients. These
directories are gitignored. Each skill is a directory whose required
`SKILL.md` contains YAML `name` and `description` fields followed by
short Markdown instructions.

<a id="contributing--coding-conventions"></a>

## Coding conventions

- Python 3.10+. Type-annotate new code; the project type-checks with
  [ty](https://docs.astral.sh/ty/).
- Formatting is enforced by `ruff` via pre-commit (``uv run --group lint pre-commit
  run -a``). The CI will reject unformatted code; running pre-commit
  locally is the easiest way to avoid surprises.
- Prefer small, well-named functions over long functions with comments
  explaining each block. Comments should explain *why*, not *what*.
- Tests live next to the thing they validate — see the File Tree Of
  FlashDreams above. Use `pytest` and prefer existing fixtures over
  hand-rolled setup. See
  [Testing](CONTRIBUTING.md#contributing--testing) for discovery and marker requirements.
- Every source file added by a contribution must include the SPDX
  header used elsewhere in the project:

```python

```

  # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0
  #
  # Licensed under the Apache License, Version 2.0 (the "License");
  # you may not use this file except in compliance with the License.
  # You may obtain a copy of the License at
  #
  # http://www.apache.org/licenses/LICENSE-2.0
  #
  # Unless required by applicable law or agreed to in writing, software
  # distributed under the License is distributed on an "AS IS" BASIS,
  # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  # See the License for the specific language governing permissions and
  # limitations under the License.

  External contributors should add their own copyright line *above* the
  NVIDIA line if they wish to be attributed; both attributions are
  retained.

<a id="contributing--testing"></a>

## Testing

Pytest collection is configured in the root `pyproject.toml`. There is
no `testpaths` setting, so `pytest` from the repo root walks the tree:

- Files named `test_*.py`. Pytest's default also collects `*_test.py`;
  don't use that name here.
- Classes named `Test*` (pytest default). Some modules use classes,
  some only module-level functions.
- Functions and methods named `test_*` (pytest default), including
  `async def test_*`.

Same test filename in different folders is fine (several
`test_application.py` files exist). Root `pyproject.toml` already sets
`--import-mode=importlib` so pytest keeps them separate. Do not remove
that flag.

A `test_*.py` next to the code it validates is collected with no CI
change. The root `pyproject.toml` skips `parity_check`,
`parity_check_v2`, `baseline_fastvideo`, and `baseline_lightx2v` via
`norecursedirs`.

Every test function must have a **CI tier marker**. A pytest plugin
(`flashdreams._pytest_plugins.marker_enforcement`) enforces this at
collection time: tests without a marker are rejected, as are tests with
both `ci_cpu` and `ci_gpu`. `manual` may be combined with a module-level
`ci_cpu` or `ci_gpu` marker to opt one test out of CI; `manual` takes
precedence.

| Marker | When to use | CI runner |
| --- | --- | --- |
| `@pytest.mark.ci_cpu` | Pure CPU logic, no GPU or `libGL` needed | CPU runner |
| `@pytest.mark.ci_gpu` | Needs CUDA, `libGL`, or transitive `cv2` | GPU runner (RTX Pro 6000) |
| `@pytest.mark.manual` | Heavy (OOM risk), flaky, needs credentials or large downloads | Not run in CI |

Use a module-level `pytestmark` when every test in a file shares the
same marker:

```python

import pytest

pytestmark = pytest.mark.ci_cpu

```

Use per-function markers when tests in the same file have different
tiers:

```python

@pytest.mark.ci_cpu
def test_basic_math(): ...

@pytest.mark.ci_gpu
def test_cudagraph_path(): ...

```

Running tests locally:

```bash

uv run --group test pytest -m ci_cpu          # CPU-safe tests only
uv run --group test pytest -m ci_gpu          # GPU tests only (needs CUDA)
uv run --group test pytest -m "not manual"    # everything that runs in CI
uv run --group test pytest                    # all tests including manual

```

The helper scripts accept an optional test path:

```bash

./tests/run_tests_local.sh [target]
./tests/run_tests_docker.sh [target]

```

The local script runs tests outside a container. The Docker script creates a
CUDA development container and then calls the local script. Set
`FLASHDREAMS_TEST_IMAGE` to change the container image. The Docker script
also accepts `FLASHDREAMS_UV_CACHE_DIR`, `FLASHDREAMS_HF_CACHE_DIR`,
`FLASHDREAMS_CACHE_DIR`, and `FLASHDREAMS_TRITON_CACHE_DIR` for persistent
host caches.

<a id="contributing--dependency-version-bounds"></a>

## Dependency version bounds

The `flashdreams/pyproject.toml` declares minimum version bounds for all
runtime dependencies. These bounds reflect the oldest versions we believe
are compatible based on API analysis.

**CI tests run against the pinned versions in `uv.lock`**, not against
the declared minimums. This means:

- We guarantee correctness at the locked versions.
- We expect the package to work at the declared minimum bounds, but do
  not continuously validate this in CI.
- If you encounter breakage with a version that satisfies the declared
  bounds but differs from the lock file, please
  [open an issue](https://github.com/NVIDIA/flashdreams/issues). We will
  either fix compatibility or bump the bound in `pyproject.toml`.

<a id="contributing--working-with-a-single-integration-package"></a>

## Working with a single integration package

The workspace contains many integration packages under `integrations_v2/`.
A full `uv sync` installs dependencies for *all* of them. If you only
need one (e.g. you're working on `omnidreams`), use the distribution
package name with `--package` to sync only that package's dependencies:

```bash

# Only install omnidreams + its deps (skips unrelated heavy packages)
uv sync --package flashdreams-omnidreams --extra dev

# Run a script/test from that integration only
uv run --package flashdreams-omnidreams pytest integrations_v2/omnidreams/tests/ -m ci_gpu

```

This avoids pulling in (and compiling) dependencies that other
integrations require but yours does not, further reducing setup time.

Available integration packages:

| Path | `uv --package` name |
| --- | --- |
| `integrations_v2/causal_forcing` | `flashdreams-causal-forcing` |
| `integrations_v2/color_fade` | `flashdreams-color-fade` |
| `integrations_v2/cosmos_predict2` | `flashdreams-cosmos-predict2` |
| `integrations_v2/fastvideo_causal_wan22` | `flashdreams-fastvideo-causal-wan22` |
| `integrations_v2/flashvsr` | `flashdreams-flashvsr` |
| `integrations_v2/hy_worldplay` | `flashdreams-hy-worldplay` |
| `integrations_v2/imgui_ui_demo` | `flashdreams-imgui-ui-demo` |
| `integrations_v2/lingbot` | `flashdreams-lingbot` |
| `integrations_v2/null_model` | `flashdreams-null-model` |
| `integrations_v2/omnidreams` | `flashdreams-omnidreams` |
| `integrations_v2/red_screen` | `flashdreams-red-screen` |
| `integrations_v2/sana_wm` | `flashdreams-sana-wm` |
| `integrations_v2/self_forcing` | `flashdreams-self-forcing` |
| `integrations_v2/slangpy_ui_demo` | `flashdreams-slangpy-ui-demo` |
| `integrations_v2/swiftvr` | `flashdreams-swiftvr` |
| `integrations_v2/wan21` | `flashdreams-wan21` |
| `integrations_v2/wan22` | `flashdreams-wan22` |
| `integrations_v2/waypoint` | `flashdreams-waypoint` |

The nested `integrations_v2/omnidreams/impl/ludus-renderer` workspace package is
named `ludus-renderer` and is installed as part of Omnidreams workflows
that need it.

<a id="contributing--versioning-and-publishing"></a>

## Versioning and publishing

`flashdreams/flashdreams/_version.py` is the version source of truth.
`.github/scripts/sync_version.py` copies it to workspace package metadata;
the `sync-version` pre-commit hook checks that they agree.

To prepare a release, change `__version__`, run the sync script until it
makes no more changes, and include the updated package metadata:

```bash

uv run --no-sync .github/scripts/sync_version.py

```

Only the core `flashdreams` wheel is published to PyPI. Applications,
integrations, and `ludus-renderer` remain workspace packages installed from a
repository checkout. After required CPU and GPU jobs pass on `main`, CI
publishes a core version that is not already on PyPI.

<a id="contributing--licensing-of-contributions"></a>

## Licensing of contributions

By submitting a pull request to this repository, you agree that your
contribution is licensed under the
[Apache License, Version 2.0](https://github.com/NVIDIA/flashdreams/blob/main/LICENSE), the same license under which
FlashDreams is distributed. The DCO sign-off described above is your
attestation that you have the right to make that grant.

Third-party code (i.e. code you did not write yourself, but that you
have the right to redistribute under a compatible license) may be
contributed only if:

1. its license is compatible with Apache-2.0;
2. its origin and license are clearly recorded in
   [REUSE.toml](https://github.com/NVIDIA/flashdreams/blob/main/REUSE.toml) and
   [THIRD-PARTY-NOTICES](https://github.com/NVIDIA/flashdreams/blob/main/THIRD-PARTY-NOTICES);
3. its files retain whatever attribution headers the upstream license
   requires.

If you are not sure whether something is contributable, please ask in
an issue before sending the code — it's much easier to sort out
upfront.

<a id="contributing--reporting-issues"></a>

## Reporting issues

Use [GitHub Issues](https://github.com/NVIDIA/flashdreams/issues) to report
functional defects and to request improvements. Please do not include
confidential or customer information.

Do not file security vulnerabilities as public issues. Follow the coordinated
disclosure process in
[SECURITY.md](https://github.com/NVIDIA/flashdreams/blob/main/SECURITY.md).

<a id="contributing--code-of-conduct"></a>

## Code of Conduct

This project follows the
[Code of Conduct](https://github.com/NVIDIA/flashdreams/blob/main/CODE_OF_CONDUCT.md).
By participating in this project — including issues, discussions, and
pull requests — you agree to abide by it. Please report concerns to the
maintainers via the address listed in the Code of Conduct.

---

Thanks again for contributing. The project is more useful, more correct,
and more interesting because outside contributors take the time to send
their work upstream. We appreciate it.
