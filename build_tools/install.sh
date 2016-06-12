#!/bin/bash
set -e

export CC=gcc
export CXX=g++

# we will use conda as our testing environment
deactivate

# setup conda
if [[ "$TRAVIS_PYTHON_VERSION" == "2.7" ]]; then
    wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh;
else
    https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
fi

bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a

# install dependencies
conda create -q -n testvenv python=$TRAVIS_PYTHON_VERSION numpy scipy cython scikit-learn pytest
source activate testvenv
pip install pytest-xdist

# if we also need to generate a coverage report
if [[ "$COVERAGE" == "true" ]]; then
    #pip install pytest-cov python-coveralls coverage==3.7.1
    pip install pytest-cov coverage coveralls
fi

# flake8
if [[ "$TEST_MODE" == "FLAKE8" ]]; then
    pip install pytest-flake8
fi

# install package
python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python -c "import sklearn; print('sklearn %s' % sklearn.__version__)"
python setup.py develop
