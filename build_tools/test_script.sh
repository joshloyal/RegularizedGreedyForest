#!/bin/bash
set -e

# run tests
if [[ "$COVERAGE" == "true" ]]; then
    py.test --cov=rgforest --cov-report=term --cov-report=html --cov-fail-under=85 -s -v rgforest
elif [[ "$TESTMODE" == "FLAKE8" ]]; then
    py.test --flake8 -v rgforest
else
    py.test -s -v rgforest
fi
