# Everyday Causal Inference — Data and code repository

<p><em>Learn how to find causal answers to your everyday questions using R and Python.</em></p>

## About the book

At [everydaycausal.com](https://www.everydaycausal.com) you will learn how to estimate, test, and explain causal impacts. You don't need any specific background to master the ideas there. Every concept is broken down step-by-step, with relatable examples from e-commerce, fintech, and digital businesses.

## About this repository

You can read the book without running any code, but following along with the exercises is the best way to learn. To do that, you'll need *R* or *Python* installed.

### Data

All datasets used in the book are available in the [data folder](https://github.com/RobsonTigre/everyday-ci/tree/main/data). You can either:

- **Download manually** from the folder above, or
- **Load directly** in your code by passing the raw URL to `read.csv()` in R or `pd.read_csv()` in Python:

```
https://raw.githubusercontent.com/RobsonTigre/everyday-ci/main/data/<filename>.csv
```

### Setup

To install every package used across the book's code examples in one go:

**Python** (tested with 3.9+):

```bash
pip install -r requirements.txt
```

**R**:

```bash
Rscript install.R
```

or run `source("install.R")` from an R session. Each script also lists its own packages in a comment at the top, so you can install per chapter instead.

## Companion AI plugin: Everyday Causal Skills

The book has an optional companion, [everyday-causal-skills](https://github.com/RobsonTigre/everyday-causal-skills) — a free plugin that gives your AI coding agent (Claude Code, Gemini CLI, GitHub Copilot CLI, Codex CLI, or Cursor) the workflow taught in the book: describe your causal question in plain language, get a method recommendation, check the assumptions, write the analysis in R or Python, stress-test the results, and compile a report.

You don't need it to follow the book. But if you want extra hands-on practice — for example, `/causal-exercises` generates practice problems with simulated data and known ground truth — head to the [everyday-causal-skills repository](https://github.com/RobsonTigre/everyday-causal-skills) for installation instructions and worked examples.

## Contribute and stay updated

- ⭐ **Star this repo** to signal interest and get updates.
- 🔔 **Follow the [author](https://www.linkedin.com/in/robson-tigre/)** for weekly posts on causal inference.
- 🌐 **Contact:** Reach me through my [website](https://www.robsontigre.com/) or socials.
- ✍️ **Read more:** Longer-form posts on [Substack](https://robsontigre.substack.com/) and [Medium](https://medium.com/@robson.tigre0).
- 📬 **[Subscribe](https://tally.so/r/0Q7z7P)** to be notified when new content is released.
- 🚩 **Feedback:** Found an error or have a suggestion? [Submit feedback here](https://tally.so/r/obbx0V).

## Citation

If you use this material in your work, please cite:

*Tigre, Robson. Everyday Causal Inference: How to estimate, test, and explain impacts with R and Python. https://www.everydaycausal.com/*

## Legal notice

Copyright © 2025 by Robson Tigre. All rights reserved. You are welcome to clone this repository and run the code and data for personal learning, but not to redistribute the material or use it to train AI systems. See [LICENSE.md](LICENSE.md) for the full terms and the [legal notice](https://www.everydaycausal.com/legal.html) for details.
