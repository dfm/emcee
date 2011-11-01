all: acor/_acor.so

acor/_acor.so: acor/acor.h acor/*.cpp acor/acor.i
	swig -Wall -c++ -python -o acor/acor.cpp acor/acor.i
	python setup.py build_ext --inplace

clean:
	rm -f acor/_acor.so 