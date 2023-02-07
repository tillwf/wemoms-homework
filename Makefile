.PHONY: clean requirements dataset build-features merge-features train predictions tests

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = wemoms_homework
PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e .

## Make Dataset
dataset:
	$(PYTHON_INTERPRETER) -m wemoms_homework make-dataset

build_features:
	$(PYTHON_INTERPRETER) -m wemoms_homework build-features

merge_features:
	$(PYTHON_INTERPRETER) -m wemoms_homework merge-features

train:
	$(PYTHON_INTERPRETER) -m wemoms_homework train-model

predictions:
	$(PYTHON_INTERPRETER) -m wemoms_homework make-predictions

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Test python environment is setup correctly
tests:
	$(PYTHON_INTERPRETER) -m pytest tests
