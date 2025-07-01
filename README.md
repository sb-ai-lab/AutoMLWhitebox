## AutoWoE library
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/AutoWoE)](https://pypi.org/project/AutoWoE)
[![PyPI - Version](https://img.shields.io/pypi/v/AutoWoE)](https://pypi.org/project/AutoWoE)
![pypi - Downloads](https://img.shields.io/pypi/dm/AutoWoE?color=green&label=PyPI%20downloads&logo=pypi&logoColor=green)
[![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/sb-ai-lab/AutoMLWhitebox/CI.yml)](https://github.com/sb-ai-lab/AutoMLWhitebox/actions/workflows/CI.yml?query=branch%3Amaster)

This is the repository for **AutoWoE** library, developed by LightAutoML group. This library can be used for automatic creation of interpretable ML model based on feature binning, WoE features transformation, feature selection and Logistic Regression.

**Authors:** Vakhrushev Anton, Grigorii Penkin, Alexander Kirilin

**Library setup** can be done by one of three scenarios below:

1. Installation from PyPI:
```bash
pip install autowoe
```
2. Installation from source code

First of all you need to install [git](https://git-scm.com/downloads) and [poetry](https://python-poetry.org/docs/#installation).

```bash

# Load WhiteBox source code
git clone https://github.com/AILab-MLTools/AutoMLWhitebox.git

cd AutoMLWhiteBox/

# !!!Choose only one item!!!

# 1. Recommended: Create virtual environment inside your project directory
poetry config virtualenvs.in-project true

# 2. Global installation: Don't create virtual environment
poetry config virtualenvs.create false --local

# For more information read poetry docs

# Install WhiteBox
poetry install

```


**Usage tutorials** are in Jupyter notebooks in the repository root. For **parameters description** take a look at `parameters_info.md`.

**Bugs / Questions / Suggestions:**
- Seek prompt advice in [Telegram group](https://t.me/joinchat/sp8P7sdAqaU0YmRi).
- Open bug reports and feature requests on GitHub [issues](https://github.com/sb-ai-lab/AutoMLWhitebox/issues).
- Also follow our [Telegram channel](https://t.me/lightautoml)
