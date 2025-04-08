# A NOMAD optimizer plugin for ropt
This package installs a plugin for the [ropt](https://github.com/tno-ropt/ropt)
ensemble optimizer package, giving access to algorithms from the
[NOMAD](https://www.gerad.ca/en/software/nomad/) optimization package.

See also the online [`ropt`](https://tno-ropt.github.io/ropt/) and
[`ropt-nomad`](https://tno-ropt.github.io/ropt-nomad/) manuals for more
information.


## Dependencies
This code has been tested with Python version 3.11 on linux.

The backend requires the [NOMAD](https://www.gerad.ca/en/software/nomad/)
optimizer, which will be installed as a dependency from PyPI.


## Installation
From PyPI:
```bash
pip install ropt-nomad
```

## Development
The `ropt-nomad` source distribution can be found on
[GitHub](https://github.com/tno-ropt/ropt-nomad). It uses a standard
`pyproject.toml` file, which contains build information and configuration
settings for various tools. A development environment can be set up with
compatible tools of your choice.

The `ropt-nomad` package uses [ruff](https://docs.astral.sh/ruff/) (for
formatting and linting), [mypy](https://www.mypy-lang.org/) (for static typing),
and [pytest](https://docs.pytest.org/en/stable/) (for running the test suite).
