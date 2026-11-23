# Publishing `pytspredit` to PyPI

This is the same flow already validated for **pydaltoolbox**. The one-time account
setup (PyPI/TestPyPI accounts, 2FA, API token) is **already done** — Eduardo's
account-wide token `daldeploy` lives in `C:\Users\eduar\.pypirc` and works for every
DAL package. For the from-zero account walkthrough, see
`C:\Git\dal\pydaltoolbox\develop\deploy.md` (Part 0).

Key facts (same as CRAN, but): **no human review** (the upload is live immediately)
and **versions are immutable** (once `0.1.0` is uploaded, the next must be `0.1.1`,
`0.2.0`, ...). It is **beta until `1.0.0`**.

---

## Release checklist (every release)

From the project root (`c:/Git/dal/pytspredit`):

```bash
# 1. bump version in pyproject.toml if not the first release (e.g. 0.1.0 -> 0.1.1)

# 2. quality gate — must be green
ruff check .
python -m pytest
python notebooks/conversor.py        # all notebooks must execute

# 3. clean + build
#   Git Bash:    rm -rf dist build src/*.egg-info
#   PowerShell:  Remove-Item -Recurse -Force dist, build, src/*.egg-info -ErrorAction SilentlyContinue
python -m build

# 4. validate metadata (like a mini `R CMD check`)
python -m twine check dist/*          # both files must report PASSED
```

This produces in `dist/`:

- `pytspredit-0.1.0.tar.gz` — source distribution (sdist; dev material excluded).
- `pytspredit-0.1.0-py3-none-any.whl` — wheel (what `pip` installs).

---

## Rehearse on TestPyPI (recommended the first time)

```bash
python -m twine upload --repository testpypi dist/*
```

Verify it installs in a throwaway venv (`--extra-index-url` lets pip pull
numpy/pandas/sklearn/statsmodels/pydaltoolbox from the real PyPI, since TestPyPI
does not host them):

```bash
python -m venv /tmp/try && . /tmp/try/Scripts/activate      # Windows Git Bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pytspredit
python -c "import pytspredit; print(pytspredit.__version__)"
deactivate
```

---

## Publish for real

```bash
python -m twine upload dist/*
```

With the `[pypi]` section filled in `~/.pypirc`, twine does not prompt. Within a
minute `pip install pytspredit` works for everyone and the project page is
`https://pypi.org/project/pytspredit/`.

> **Who runs it:** Eduardo does (or explicitly authorizes) the upload. The first
> (and only v1) publication happens at the **end of the final phase**, after a single
> GitHub review of the completed package.

---

## Gotchas

- **Name must be free** — `pytspredit` is claimed by whoever publishes first.
- **Versions are immutable** — you cannot re-upload `0.1.0`, even after deleting it.
- **Dependency on pydaltoolbox** — it is a normal PyPI dependency (`pydaltoolbox>=0.1.0`),
  so `pip install pytspredit` pulls it automatically. (TestPyPI needs the
  `--extra-index-url` above to find it.)
- **Keep the token secret** — `.pypirc` holds a live credential.
