.PHONY: setup venv run clean

# Configurations
VENV_NAME ?= venv
PYTHON = ${VENV_NAME}/bin/python
PIP = ${VENV_NAME}/bin/pip

# Default command
all: setup

# Setup virtual environment
setup: venv
	@echo "Installing Python packages..."
	@${PIP} install -r requirements.txt

# Create virtual environment
venv: $(VENV_NAME)/bin/activate

$(VENV_NAME)/bin/activate: requirements.txt
	@test -d $(VENV_NAME) || python3 -m venv $(VENV_NAME)
	@${PIP} install --upgrade pip setuptools wheel

# Run the main Python script
run: venv
	@echo "Running app..."
	@${PYTHON} app.py

# Clean up the project (remove virtual environment)
clean:
	@echo "Cleaning up..."
	@rm -rf $(VENV_NAME)
	@find . -type f -name '*.pyc' -delete
	@find . -type d -name '__pycache__' -delete

# Help command to display makefile command help
help:
	@echo "Available commands:"
	@echo "   make setup        - Setup the project and install dependencies"
	@echo "   make run          - Run the main application"
	@echo "   with virtual environment"
	@echo "   make clean        - Clean the project directory"
