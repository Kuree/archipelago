#!/bin/sh
set -e

# thunder
python setup.py sdist upload -r pypi
cd ..
