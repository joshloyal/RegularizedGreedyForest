#!/bin/bash
set -e

# run tests
if [[ "$COVERAGE" == "true" ]]; then
    py.test --cov=rgforest --cov-report=term --cov-report=html -s rgforest
elif [[ "$TESTMODE" == "FLAKE8" ]]; then
    py.test --flake8 rgforest
else
    py.test -s rgforest
fi
