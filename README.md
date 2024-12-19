[![python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=ffd343)](https://docs.python.org/3.11/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
<!-- omit from toc -->
# Neuroevolution: Flappy Bird
A Pygame simulation of Flappy Bird, played by AI trained using neuroevolution.

<!-- omit from toc -->
## Table of Contents

- [Installing Dependencies](#installing-dependencies)
- [Running the Application](#running-the-application)
- [Linting and Formatting](#linting-and-formatting)

## Installing Dependencies

Install the required dependencies using [pipenv](https://github.com/pypa/pipenv):

    pipenv install
    pipenv install --dev

## Running the Application

Enter the virtual environment with

    pipenv shell

The application can then be started by running

    python main.py

This will open a Pygame window and begin the training. The application can be exited by closing the window.

## Linting and Formatting
This library uses `ruff` for linting and formatting.
This is configured in `pyproject.toml`.

To check the code for linting errors:

    python -m ruff check .

To format the code:

    python -m ruff format .
