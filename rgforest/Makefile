all:
	python setup.py build_ext --inplace

.PHONY: clean test

clean:
	-rm -r build *.cpp *.so

test:
	-py.test tests/ -s
