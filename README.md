# Quant Finance Notes

A lightweight notebook/code collection documenting my learning in quantitative finance.  
Single **virtual environment** at the repo root; each topic lives in its own subfolder (e.g., _Geometric Brownian Motion_, _Monte Carlo_).

> Educational only — not investment advice.

## Quickstart (one venv for everything)

```bash
# clone
git clone https://github.com/<you>/QuantFinanceNotes.git
cd QuantFinanceNotes

# create & activate a single env
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# install all dependencies
pip install -r requirements.txt

# register a Jupyter kernel (so notebooks pick this env)
python -m ipykernel install --user --name qfn --display-name "qfn (Quant Finance)"

# launch notebooks
jupyter lab
```
