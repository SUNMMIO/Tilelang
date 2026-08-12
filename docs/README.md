# TileLang-Mesh Documentation

The documentation was built upon [Sphinx](https://www.sphinx-doc.org/en/master/).

## Dependencies

Run the following command in this directory to install dependencies first:

```bash
python -m pip install -r docs/requirements.txt
```

## Build the Documentation

From the repository root, build with warnings treated as errors:

```bash
sphinx-build -W --keep-going -b html docs docs/_build/html
```

## View the Documentation

Run the following command to start a simple HTTP server:

```bash
cd docs/_build/html
python3 -m http.server
```

Then you can view the documentation in your browser at `http://localhost:8000` (the port can be customized by appending `-p PORT_NUMBER` in the python command above).
