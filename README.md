# Machine Learning LLM Wiki Example

An example repository implementing Andrej Karpathy’s LLM Wiki pattern using machine learning notes as the knowledge domain.

The repository separates raw source material from polished wiki pages, with Claude Code maintaining the wiki based on the instructions in `CLAUDE.md`.

## Start Here

- [Wiki Index](wiki/index.md)

## What This Demonstrates

- Keeping source material in `raw/`
- Generating structured Markdown notes in `wiki/`
- Maintaining a table of contents in `wiki/index.md`
- Tracking updates in `wiki/log.md`
- Using `CLAUDE.md` to guide Claude Code

## Repository Structure

- `raw/` contains source material.
- `wiki/` contains polished Markdown pages.
- `wiki/index.md` is the table of contents.
- `wiki/log.md` records updates.
- `CLAUDE.md` contains instructions for Claude Code.

## Example Domain

This example uses machine learning topics such as:

- Regression
- Decision Trees
- Random Forest
- Ensemble Methods
- Clustering
- Feature Importance

## Workflow

1. Add source material to `raw/`.
2. Ask Claude Code to ingest the new material.
3. Claude Code creates or updates pages in `wiki/`.
4. `wiki/index.md` and `wiki/log.md` are kept updated.

## Inspiration

- Andrej Karpathy’s LLM Wiki idea: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
- Obsidian: https://obsidian.md/