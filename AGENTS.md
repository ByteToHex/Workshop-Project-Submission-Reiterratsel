# Notebook / Jupytext Pairing Convention

## Python Environment

- Always use Anaconda environment named `env` for Python execution in this repository.
- Treat `env` as the required project environment.
- Explicitly use the interpreter from `C:\ProgramData\anaconda3\envs\env\python.exe` when running Python scripts.
- Explicitly use the matching Streamlit executable from `C:\ProgramData\anaconda3\envs\env\Scripts\streamlit.exe` when running Streamlit apps.
- Assume this environment is the project-standard Python `3.14` environment.
- Do not default to `base`, `env310`, `env312`, or any system Python unless the user explicitly says otherwise.

This repository uses Jupytext-paired notebooks under `Data/Data_Company/5_Model_KG/DesignRef/RS/Day_1/Workshop/`.

## Scope

Applies to files matching:

- `Data/Data_Company/5_Model_KG/DesignRef/RS/Day_1/Workshop/**/*.ipynb`
- `Data/Data_Company/5_Model_KG/DesignRef/RS/Day_1/Workshop/**/*.py`

## Rule

Always read the `.py` Jupytext sister file. Never open, read, search, or reason about the `.ipynb` file for any purpose.

Treat the `.py` file as the canonical source, even if the notebook may appear out of sync.

## Pairing Convention

Each notebook has a same-directory paired file:

- `<name>.ipynb`
- `<name>.py` in Jupytext `percent` format

The `.py` file typically starts with a Jupytext header such as:

```python
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---
```

## Banned Reasoning

Do not use any of these to justify touching `.ipynb` files:

- Checking notebook outputs or completion state
- Looking for rendered results
- Verifying whether `.py` and `.ipynb` are in sync
- Inspecting notebook metadata

Use the `.py` file only.

## Why

- `.ipynb` files contain JSON, embedded outputs, and metadata noise.
- `.py` Jupytext files are plain text and easy to read, search, and edit.
- The `.py` file is the canonical editable source by repository convention.

## Folder Shape

Typical layout:

```text
Workshop/
  <Subfolder>/
    <name>.ipynb
    <name>.py
```

Example:

`Data/Data_Company/5_Model_KG/DesignRef/RS/Day_1/Workshop/enhancing_rag_with_graph(Gemini,MD).py`

## Searching

When locating notebook content, search only `**/*.py` paths in this subtree. Do not grep or inspect `.ipynb` files.
