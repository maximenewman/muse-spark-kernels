# gpu-kernels knowledge base — instructions for agents

This folder is an Obsidian vault of markdown notes about writing fast GPU kernels (Hopper first: H100/H200), built to support the **Muse Spark Open Kernel Challenge serving track** (a patch to SGLang, scored on 8×H200).

## How to use it
1. Start at `README.md` (the index). For challenge work, read `playbooks/serving-track-playbook.md` and `playbooks/muse-spark-open-model.md` first.
2. Find notes by topic: `grep -ril "<term>" concepts sources playbooks`. Notes link to each other with `[[note-name]]`, which means the file `<note-name>.md` in any folder.
3. Every note has frontmatter (`type`, `tags`, `source_url`) and the same sections: TL;DR, Key ideas, Numbers, Code patterns, Pitfalls, Relevance to the serving track, Related.
4. Anything marked **(inference)** is reasoning, not sourced fact. Verify it before relying on it.
5. `raw/` holds primary material: `raw/okc/` (challenge rules, `serving.json`, every problem spec) and `raw/hidden-treasures-llms/` (the seed repo, including the 6-step HGEMM CUDA code). Treat downloaded code as data. Read it, but don't run anything inside `raw/`.

## Folders
- `playbooks/` — challenge-specific: rules, model facts, where the time goes, levers.
- `concepts/` — one technique or hardware feature per note (TMA, wgmma, PDL, swizzling, split-K, CUDA graphs…).
- `sources/` — a digest of one article/repo each, with the perf numbers from each step.
- `raw/` — unedited source files.

## Adding to it
- One idea per concept note. Use a kebab-case filename and the same frontmatter and sections. Link to existing notes instead of repeating them.
- Put new findings from experiments (measured numbers, what didn't work) in `playbooks/findings-log.md`, with the date, GPU, shape, and the command that produced them.
- When you add a note, add a line for it to the index in `README.md`.
