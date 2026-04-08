# Project

Underfit is an open-source model reporting dashboard for tracking experiments, metrics, and artifacts. It serves a similar role to Weights & Biases or Tensorboard, with a focus on transparent, self-hostable reporting. This repository contains the python client SDK, it targets Python 3.9+ and should remain easy to inspect for automatic API docs generation.

# Contributing

## How to contribute

- Keep code clean and concise. Do not add comments unless the logic is non-obvious.
- Avoid splitting statements across multiple lines without a readability benefit. If it fits under 160 characters, keep it on one line.
- Prefer minimal, clear abstractions over clever ones.
- Tests are located in a top-level `tests` folder.

## Docstrings

- Treat docstrings as a primary API contract for generated docs.
- Prefer Google-style docstrings for public modules, classes, methods, and functions.
- Follow PEP 257 spacing: no blank line after function or method docstrings, and keep a blank line between class docstrings and methods.
- Keep private/internal helpers undocumented unless behavior is non-obvious.
- Write a concise one-line summary first, then optional detail paragraphs.
- Use imperative tone in summaries (for example: "Initialize a new run.").
- Rely on Python type hints for types; do not duplicate complex type information in prose.
- Document arguments in `Args:` using parameter names that exactly match the function signature.
- Document return values in `Returns:` only when the return is non-trivial or important for callers.
- Document raised exceptions in `Raises:` when callers should handle them.
- Add `Examples:` for user-facing APIs when usage is not obvious.
- Keep lines readable and consistent; prefer short paragraphs and simple sentences.
- Keep wording stable and factual; avoid implementation details that may churn.
- If cross-references are needed, use fully qualified names in text (for example `underfit.Run`).

## Committing changes

- Always run the linter, type checker, and tests before committing.
- Run the linter with `ruff check .`, the typechecker with `ty check .`, and the tests with `pytest .`.
- Use single-line commit messages in plain English.
- Do not use conventional commit prefixes or add signatures (e.g. Co-Authored By)
- Run `git add` and `git commit` sequentially (or in one chained command), not in parallel, to avoid `.git/index.lock` conflicts.
