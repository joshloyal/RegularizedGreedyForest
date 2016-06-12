#!/bin/bash
set -e

# run tests
if [[ "$COVERAGE" == "true" ]]; then
    py.test --cov=rgforest --cov-report=term --cov-report=html -s rgforest
else
    py.test -s rgforest
fi
