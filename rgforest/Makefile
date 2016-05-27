all:
	python setup.py build_ext --inplace

.PHONY: clean test
clean:
	-rm -r build rgf.cpp rgf.so

test:
	-py.test test/ -s
