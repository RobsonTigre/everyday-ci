# Everyday Causal Inference — Data and Code Repository

> Learn how to find causal answers to your everyday questions using R and Python.

## About the book

At [everydaycausal.com](https://www.everydaycausal.com) you will learn how to estimate, test, and explain causal impacts. You don't need any specific background to master the ideas there. Every concept is broken down step-by-step, with relatable examples from e-commerce, fintech, and digital businesses.

## About this repository

You can read the book without running any code, but following along with the exercises is the best way to learn. To do that, you'll need *R* or *Python* installed.

### Data

All datasets used in the book are available in the [data folder](https://github.com/RobsonTigre/everyday-ci/tree/main/data). You can either:

- **Download manually** from the folder above, or
- **Load directly** in your code using the raw URL pattern:

```
https://raw.githubusercontent.com/RobsonTigre/everyday-ci/main/data/<filename>.csv
```

For example, to load `advertising_data.csv`:

```r
# R
df <- read.csv("https://raw.githubusercontent.com/RobsonTigre/everyday-ci/main/data/advertising_data.csv")
```

```python
# Python
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/RobsonTigre/everyday-ci/main/data/advertising_data.csv")
```

## Stay updated

- ⭐ **Star this repo** to signal interest and get updates.
- 🔔 **Follow the [author on LinkedIn](https://www.linkedin.com/in/robson-tigre/)** for new chapters and announcements.
- 📬 **[Subscribe here](https://tally.so/r/0Q7z7P)** to be notified when new content is released.

## Contribute

Found an error or have a suggestion? [Submit feedback here](https://tally.so/r/obbx0V).

## Citation

If you use this material in your work, please cite:

> Tigre, Robson. *Everyday Causal Inference: How to estimate, test, and explain impacts with R and Python.* https://www.everydaycausal.com/

## Legal notice

Copyright © 2025 by Robson Tigre. All rights reserved. You may read, share official links, and cite short excerpts for learning purposes, provided you credit the source. However, you may not reproduce, redistribute, or use any part of this book (including text and code) to train AI systems without explicit permission.

This content is for educational purposes only and does not constitute professional advice. All code is provided "as is," without warranty. The author disclaims all liability for outcomes based on this material. The full legal notice is available [here](https://www.everydaycausal.com/legal.html).
