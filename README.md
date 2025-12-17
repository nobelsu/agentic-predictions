# Agentic Predictions

An autonomous, self-optimizing **agentic prediction pipeline** for evaluating startup success, built using the **Model Context Protocol (MCP)** and the `mcp-agent` framework.

This repository contains agent definitions, schemas, utilities, and sample data to run and extend multi-agent prediction workflows.

## Table of contents

* [Overview](#overview)
* [Features](#features)
* [Repository structure](#repository-structure)
* [Getting started (uv)](#getting-started-uv)
* [Configuration](#configuration)
* [Running the project](#running-the-project)
* [Development notes](#development-notes)
* [Contributing](#contributing)
* [License](#license)

## Overview

This project explores **agentic reasoning** for prediction tasks. Multiple agents collaborate through MCP to gather context, reason over structured inputs, and produce predictions (e.g. startup success likelihood). The system is designed to be modular, extensible, and easy to experiment with.

## Features

* 🧠 **Multi-agent architecture** using `mcp-agent`
* 🔌 **MCP-compatible** configuration and tool exposure
* 📊 Example datasets and schemas for prediction tasks
* 🧪 Scripted experiment runner for fast iteration
* ⚙️ Designed for extension with new agents, tools, or data sources

## Repository structure

```
.
├── agents/                  # Agent definitions and workflows
├── data/                    # Datasets used by experiments
├── schema/                  # Input / output schemas
├── utils/                   # Shared utilities
├── mcp_agent.config.yaml    # MCP configuration
├── script.py                # Main entrypoint
├── values.csv               # Example dataset
├── pyproject.toml           # Project metadata (uv-compatible)
├── uv.lock                  # Locked dependencies
└── LICENSE                  # MIT License
```

## Getting started (uv)

This project uses **[`uv`](https://github.com/astral-sh/uv)** for dependency management and environment setup.

### Prerequisites

* Python **3.10+**
* `uv` installed

Install `uv` if you don’t already have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install dependencies

Clone the repo and sync dependencies:

```bash
git clone https://github.com/nobelsu/agentic-predictions.git
cd agentic-predictions

uv sync
```

This will:

* Create an isolated virtual environment
* Install all dependencies defined in `pyproject.toml`
* Use `uv.lock` for reproducible builds

### Activate the environment (optional)

```bash
source .venv/bin/activate
```

> Not strictly required — `uv run` works without manual activation.

## Configuration

### MCP configuration

Edit `mcp_agent.config.yaml` to configure:

* MCP servers and transports
* Tool exposure
* Agent runtime settings

### Environment variables

If using hosted LLMs, set your API key(s):

```bash
export OPENAI_API_KEY=your_key_here
```

(or equivalent for your provider)

## Running the project

The main entrypoint is `script.py`.

Run it using `uv`:

```bash
uv run python script.py
```

If the script accepts flags or a config path:

```bash
uv run python script.py --config mcp_agent.config.yaml
```

Check the top of `script.py` for supported arguments and defaults.

## Development notes

* Add new agents under `agents/` and register them in workflows.
* Keep schemas in `schema/` and validate early.
* Any new MCP tools or endpoints should be declared in `mcp_agent.config.yaml`.
* Prefer `uv add <package>` when adding dependencies:

  ```bash
  uv add pandas
  ```

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make changes with clear commits
4. Open a pull request with context and rationale

## License

MIT License. See `LICENSE` for details.
