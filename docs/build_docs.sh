#!/bin/bash

# Create required directories
mkdir -p docs
mkdir -p site

# Exit on error
set -e

# Create and activate virtual environment
python -m venv docs_venv
. docs_venv/Scripts/activate  # Changed activation path for Windows

# Install dependencies
pip install -r requirements.txt

# Build documentation
mkdocs build

# Serve documentation (optional)
if [ "$1" = "serve" ]; then
    echo "Starting documentation server..."
    mkdocs serve
fi

# Create versioned documentation (optional)
if [ "$1" = "version" ]; then
    if [ -z "$2" ]; then
        echo "Please provide a version number"
        exit 1
    fi
    echo "Creating documentation for version $2..."
    mike deploy --push --update-aliases $2 latest
fi

# Cleanup
if [ "$1" != "serve" ]; then
    deactivate
    echo "Documentation built successfully!"
    echo "You can find the static files in the 'site' directory"
fi 