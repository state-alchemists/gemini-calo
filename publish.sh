#!/bin/bash
# This script builds and publishes the package to PyPI.
# Make sure you have your PyPI API token configured for Poetry.
# You can configure it with: poetry config pypi-token.pypi YOUR_TOKEN

echo "Building the package..."
poetry build

echo "Publishing the package to PyPI..."
poetry publish
