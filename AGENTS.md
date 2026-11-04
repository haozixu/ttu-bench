# Working agreements

- Keep documentation organized by subject. Update an existing document instead of creating a phase report for each small change. Keep the README and handoff short; avoid duplicating conclusions across files.
- Public documentation must stand on tracked files and reproducible commands. Never cite ignored output directories as evidence or maintain inventories of temporary files.
- Routine experiments need a command and a concise result, not full artifacts. Raw buffers, cubins, per-case dumps and repeated reports are opt-in and disposable. Clean temporary builds, unsuccessful probes and duplicates after use.
- Never write files directly into the root of `/tmp`. First create a task-specific directory under `/tmp` (prefer `mktemp -d` or `TemporaryDirectory`) and put all temporary files and outputs inside it. Clean up that directory after use. Apply this rule to commands, scripts and documentation examples; never remove unrelated temporary files.
- Retain complete evidence only for important protocol properties, counterexamples that overturn a conclusion, or results that cannot reasonably be reproduced. Prefer a small self-contained example; retain helpers, source code, original JIT/AS fixtures and useful EoSS behavior knowledge.
- Do not silently loosen correctness checks to make a matrix pass. Distinguish software/hardware numerical disagreement from unsupported TTU functionality.
