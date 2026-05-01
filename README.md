# Machine Learning LLM Wiki

A personal knowledge base built on Andrej Karpathy's LLM Wiki pattern, using machine learning as the knowledge domain. Raw source material is ingested by Claude Code and transformed into a structured, interlinked wiki that compounds over time.

## Start Here

- [Wiki Index](wiki/index.md)
- [Update Log](wiki/log.md)

## How It Works

Instead of querying raw documents every time, Claude Code reads new sources and integrates them into persistent wiki pages — updating summaries, linking concepts, and refining existing content. The wiki grows richer with every source added.

## Repository Structure

```
machine-learning-wiki/
├── raw/          # Immutable source material — never modified
├── wiki/         # LLM-maintained knowledge base
│   ├── index.md  # Table of contents
│   └── log.md    # Chronological update history
└── CLAUDE.md     # Schema — instructs Claude how to maintain the wiki
```

## Topics Covered

- Linear Models (OLS, Ridge, Lasso, Elastic-Net, Logistic Regression)
- Decision Trees (CART, pruning, impurity criteria)
- Ensemble Methods (Gradient Boosting, Random Forests, Bagging, Stacking)
- Clustering (K-Means, DBSCAN, HDBSCAN, Hierarchical)

## Workflow

1. Clip or save source material into `raw/`
2. Tell Claude Code to ingest new files
3. Claude reads, synthesizes, and writes structured pages into `wiki/`
4. `wiki/index.md` and `wiki/log.md` are updated automatically

## Inspiration

- [Andrej Karpathy's LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
- [Obsidian](https://obsidian.md/)
- [Claude Code](https://claude.ai/code)
