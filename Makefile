.PHONY: setup venv run clean help

# Configurations
VENV_NAME ?= venv
PYTHON = ${VENV_NAME}/bin/python
PIP = uv pip

# Default command
all: setup

# Create and activate virtual environment
venv:
	@test -d $(VENV_NAME) || uv venv $(VENV_NAME)
	@echo "Virtual environment created. Activate with: source $(VENV_NAME)/bin/activate"

# Setup virtual environment and install packages
setup: venv
	@echo "Installing Python packages..."
	@. $(VENV_NAME)/bin/activate; $(PIP) install -r requirements.txt; deactivate

# Run the main Python script
run:
	@echo "Running app..."
	@. $(VENV_NAME)/bin/activate; $(PYTHON) app.py; deactivate

# Clean up the project (remove virtual environment)
clean:
	@echo "Cleaning up..."
	@rm -rf $(VENV_NAME)
	@find . -type f -name '*.pyc' -delete
	@find . -type d -name '__pycache__' -delete

# Help command to display Makefile command help
help:
	@echo "Available commands:"
	@echo "   make setup        - Setup the project and install dependencies"
	@echo "   make run          - Run the main application with virtual environment"
	@echo "   make clean        - Clean the project directory"
	@echo "   make help         - Display this help message"
