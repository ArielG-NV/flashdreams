---
name: adding-documentation
description: Add or revise FlashDreams documentation and repository Agent Skills while preserving the canonical documentation layout, concise cross-links, and skill discovery rules. Use when creating, moving, or substantially editing Markdown, README files, or skills/**/SKILL.md.
---

# Add documentation or an Agent Skill

Read [CONTRIBUTING.md](../../CONTRIBUTING.md) before editing. Its **Adding documentation** and **Adding an Agent Skill** sections are authoritative; do not duplicate those rules here.

## Documentation

Put canonical prose under `docs/src/content/docs/**` in Markdown (`.md`). Prefer a subsection in an existing topic page; create a file only for a distinct technical reference. Write general guidance in short, plain English and link to the exact existing section instead of repeating it. The only other multiline Markdown is:

- `skills/**/SKILL.md`, which must remain in `skills/`; and
- `flashdreams/README.md`, which must retain its own package introduction.

All other Markdown files are one-line links to a file and section under `docs/src/content/docs/**`. When documenting an API, distinguish the inference API, demo API, and separate v2 application protocol using the API overview linked from `CONTRIBUTING.md`.

## Agent Skills

Create the source directly at `skills/<skill-name>/SKILL.md`; never mirror a skill under `docs/src/content/docs/`. Keep the frontmatter description specific enough for discovery, keep the body short, and link to `CONTRIBUTING.md` or canonical docs for shared rules. Add the skill to the Skill Map in `AGENTS.md`.

## Validate

Run the focused checks from `CONTRIBUTING.md`, then the repository-required lint and CPU test commands. Do not stage, commit, or open a pull request unless the user asks.
