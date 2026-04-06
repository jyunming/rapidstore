---
name: Bug report
about: Something is broken or behaves unexpectedly
title: "fix: "
labels: bug
assignees: ''
---

## Describe the bug

A clear description of what is broken.

## Reproduction

```python
# Minimal script that reproduces the issue
from tqdb import Database
import numpy as np

db = Database.open("./repro", dimension=...)
```

## Expected behavior

What should happen.

## Actual behavior

What actually happens (include full traceback if applicable).

## Environment

- tqdb version (`pip show tqdb`):
- Python version:
- OS:
- Installed from: PyPI / built from source
