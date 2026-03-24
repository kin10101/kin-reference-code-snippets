# UV Setup Guide

UV is a fast, modern Python package manager and project tool written in Rust. It replaces `pip`, `venv`, and `poetry` with a single, unified interface.

## Installation

### Option 1: Using pip (if you already have Python)

```bash
    pip install uv
```

### Option 2: Standalone installer (recommended)

Download and run the installer from https://github.com/astral-sh/uv/releases or use your package manager:

**Windows (PowerShell):**
```powershell
irm https://astral-sh.github.io/uv/install.ps1 | iex
```

**macOS/Linux:**
```bash
curl -LsSf https://astral-sh.github.io/uv/install.sh | sh
```

Verify installation:
```bash
uv --version
```

## Getting Started

### Create a new project

```bash
uv init my-project
cd my-project
```

This creates a basic project structure with `pyproject.toml`.

### Create a virtual environment (if not using uv init)

```bash
uv venv
```

Activate the environment (choose your shell):

```text
# macOS/Linux (bash/zsh)
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows Command Prompt
.venv\Scripts\activate.bat
```

### Work with an existing project

If you have a project with `pyproject.toml`:

```bash
# Sync dependencies to a virtual environment
uv sync
```

Activate the environment (choose your shell):

```text
# macOS/Linux (bash/zsh)
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows Command Prompt
.venv\Scripts\activate.bat
```

## Managing Dependencies

### Add a package

```bash
uv add requests
uv add flask django
```

### Add a development dependency

```bash
uv add --dev pytest black ruff
```

### Remove a package

```bash
uv remove requests
```

### Update dependencies

```bash
uv sync  # Install/update based on lock file
uv lock  # Update lock file without installing
```

### View installed packages

```bash
uv pip list
```

## Running Commands

### Run Python scripts

```bash
uv run main.py
uv run python script.py
```

### Run commands in the virtual environment

```bash
uv run pytest
uv run black .
uv run python -m flask run
```

### Install and run a tool

```bash
uv tool run black .  # Auto-installs black in isolated environment
uv tool run pytest   # Auto-installs pytest in isolated environment
```

## Configuration (pyproject.toml)

UV uses `pyproject.toml` for configuration. Example:

```toml
[project]
name = "my-project"
version = "0.1.0"
description = "A sample project"
requires-python = ">=3.8"
dependencies = [
    "requests>=2.28.0",
    "flask>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "black>=22.0",
    "ruff>=0.1.0",
]

[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[tool.uv]
dev-dependencies = [
    "pytest",
    "black",
]
```

## Common Workflows

### Set up a new Python project

```bash
uv init my-project
cd my-project
uv add requests flask
uv add --dev pytest black ruff
uv sync
```

```text
# Then activate the environment:
# macOS/Linux (bash/zsh)
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

### Clone and install an existing project

```bash
git clone <repo>
cd <repo>
uv sync
uv run pytest
```

```text
# Activate the environment first:
# macOS/Linux (bash/zsh)
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

### Use a specific Python version

```bash
uv venv --python 3.11
uv sync
```

Ensure you have the Python version installed on your system.

### Pin a specific package version

Edit `pyproject.toml`:
```toml
dependencies = [
    "requests==2.31.0",  # Exact version
    "flask>=2.3.0,<3",   # Version range
]
```

Then run:
```bash
uv lock
uv sync
```

## Useful Commands

```bash
uv --help              # Show all available commands
uv init --help         # Help for a specific command
uv venv                # Create virtual environment
uv sync                # Install dependencies from lock file
uv lock                # Update lock file based on pyproject.toml
uv add <package>       # Add a dependency
uv remove <package>    # Remove a dependency
uv run <command>       # Run a command in the virtual environment
uv pip list            # List installed packages
uv tool run <tool>     # Run a tool (auto-installs if needed)
```

## Why UV?

- **Fast**: Rewritten in Rust, 10-100x faster than pip
- **Single tool**: Replaces pip, venv, and poetry
- **Python-aware**: Manages Python versions automatically
- **Lock files**: Deterministic dependency resolution
- **Simple**: Minimal configuration needed
- **Compatible**: Works with standard `pyproject.toml`

## Tips

- UV automatically creates a virtual environment in `.venv/` when you run most commands
- Use `uv tool run` for one-off tools to keep your project dependencies clean
- The `uv.lock` file should be committed to version control for consistent installs
- For monorepos or complex setups, UV provides workspaces support

## Troubleshooting (Windows PowerShell)

If you see this error when activating the virtual environment:

```powershell
& : File .venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled on this system.
```

PowerShell execution policy is blocking local scripts. Use one of these options:

### Option 1: Temporary (current terminal only)

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Option 2: Persistent (current user)

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Then restart PowerShell and activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

If your execution policy is controlled by your organization, use UV directly without activation:

```powershell
uv run python --version
uv run pytest
```