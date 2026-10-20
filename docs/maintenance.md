# Maintain these docs

Build with Python 3.12:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -b html -E -n -W --keep-going docs docs/_build/html
```

The GitHub docs workflow builds only the docs checkout. Scientific runtime dependencies belong in their existing manifests. Update source references and examples when the canonical interface changes.

## Read the Docs

Use the lab's existing Business organization with project and PR visibility set to Private. The initial private review uses `codex/calcium-docs-remaining-20261021`. After its documentation PR is reviewed and merged, switch the default branch to `main`, rebuild latest, and verify the rendered page and source/edit links.
