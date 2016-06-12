#!/bin/bash
set -e

# run tests
if [[ "$COVERAGE" == "true" ]]; then
    py.test --cov=rgforest --cov-report=trem --cov-report=html -s rgforest
else
    py.test -s rgforest
fi
