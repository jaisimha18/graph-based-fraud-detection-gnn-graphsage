# Contributing to Graph-Based AML Fraud Detection

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

---

## 🚀 Getting Started

### 1. Fork & Clone

```bash
git clone https://github.com/<your-username>/graph-based-fraud-detection-gnn-graphsage.git
cd graph-based-fraud-detection-gnn-graphsage
```

### 2. Set Up Development Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Create a Feature Branch

```bash
git checkout -b feat/your-feature-name
```

---

## 📝 Code Style & Conventions

### Python

- Follow **PEP 8** style guidelines
- Use **type hints** for function signatures
- Write **docstrings** for all public functions and classes
- Keep functions focused and under ~50 lines where possible

### Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Usage |
|---|---|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `refactor:` | Code change that neither fixes a bug nor adds a feature |
| `test:` | Adding or updating tests |
| `chore:` | Maintenance tasks (deps, configs) |
| `ci:` | CI/CD changes |

**Example:** `feat: add temporal edge weighting to CycleAwareSAGEConv`

### JavaScript / React (Website)

- Use functional components with hooks
- Follow existing component structure in `website/src/components/`

---

## 🔀 Pull Request Process

1. Ensure your code passes any existing tests
2. Update documentation if your change affects usage
3. Write a clear PR description explaining **what** and **why**
4. Reference any related issues (e.g., `Fixes #12`)
5. Request review from a maintainer

---

## 🐛 Reporting Issues

When reporting bugs, please include:

1. **Description:** Clear summary of the issue
2. **Steps to Reproduce:** Minimal steps to trigger the bug
3. **Expected Behavior:** What you expected to happen
4. **Actual Behavior:** What actually happened
5. **Environment:** OS, Python version, PyTorch version, GPU info

---

## 📁 Project Structure Guide

| Directory | Purpose |
|---|---|
| `src/aml_pipeline/` | Data preprocessing and feature engineering |
| `src/models/` | GNN model definitions |
| `src/training/` | Training loops and evaluation scripts |
| `src/api/` | FastAPI backend |
| `scripts/` | Utility and helper scripts |
| `data/samples/` | Sample transaction CSVs for testing |
| `docs/` | Technical documentation |
| `website/` | React frontend dashboard |

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
