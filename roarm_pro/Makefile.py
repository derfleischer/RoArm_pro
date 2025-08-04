# Makefile for RoArm Pro development

.PHONY: help install develop test clean run debug list-ports

# Default target
help:
	@echo "RoArm Pro Development Commands"
	@echo "=============================="
	@echo "make install     - Install package and dependencies"
	@echo "make develop     - Install in development mode"
	@echo "make test        - Run installation tests"
	@echo "make clean       - Clean build artifacts"
	@echo "make run         - Run the application"
	@echo "make debug       - Run in debug mode"
	@echo "make list-ports  - List available serial ports"

# Install package
install:
	pip install -r requirements.txt
	python setup.py install

# Development install
develop:
	pip install -r requirements.txt
	python setup.py develop

# Run tests
test:
	python test_installation.py

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete

# Run application
run:
	python -m roarm_pro.main

# Run in debug mode
debug:
	python -m roarm_pro.main --debug

# List serial ports
list-ports:
	python -m roarm_pro.main --list-ports

# Create directories
dirs:
	mkdir -p roarm_pro/config
	mkdir -p roarm_pro/hardware
	mkdir -p roarm_pro/motion
	mkdir -p roarm_pro/control
	mkdir -p roarm_pro/ui
	mkdir -p examples
	mkdir -p tests
	mkdir -p docs
