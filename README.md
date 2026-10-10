# Master-Thesis

`NeuralNets.py` provides `FactorScoreNetwork`, the factor-structured score
architecture. Its input is a return batch `[batch, d]`
and forward OU noising times `[batch, 1]`; its output has shape `[batch, d]`.
Call `project_orthogonal()` after optimizer updates to restore orthonormal loadings.

Use Python 3.12 with the project's dependency manifest:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-equities.txt
python -m unittest discover -s tests -v
```
