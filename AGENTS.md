## Branching and GitHub workflow

- Do not make code changes directly on the `main` branch.
- Before starting a new feature or other code change, create and switch to a new branch based on the latest appropriate base branch.
- Keep feature work and commits on that branch; do not commit directly to `main`.
- If an AI assistant is used to make changes and push them to GitHub, push the working branch to the repository rather than pushing to `main`.

## First-time setup

From the repository root, create and activate the virtual environment, then install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

When working in Jupyter, select the `.venv` Python interpreter or its `ipykernel` kernel.