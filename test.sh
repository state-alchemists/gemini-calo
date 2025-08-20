#!/bin/bash
PYTHONPATH=$PYTHONPATH:. poetry run pytest --cov=gemini_calo tests/
