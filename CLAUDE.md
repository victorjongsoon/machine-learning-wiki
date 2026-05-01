# LLM Wiki

This is a personal machine learning knowledge base maintained with Claude Code.

## Structure

- `raw/` contains source material.
- `wiki/` contains polished Markdown pages.
- `wiki/index.md` is the table of contents.
- `wiki/log.md` records updates.

## Ingest Workflow

When new material is added to `raw/`:

1. Read the source.
2. Extract useful concepts, examples, formulas, and pitfalls.
3. Create or update relevant pages in `wiki/`.
4. Update `wiki/index.md`.
5. Append a short entry to `wiki/log.md`.

## Query Workflow

When answering questions:

1. Read `wiki/index.md`.
2. Read relevant pages in `wiki/`.
3. Use `raw/` only when more source detail is needed.
4. Say clearly if the answer is not in the wiki.

## Rules

- Do not modify files in `raw/` unless asked.
- Create folders only when useful.
- Avoid copying long passages; summarize in your own words.
- Prefer concise, practical explanations with examples.