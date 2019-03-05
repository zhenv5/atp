CXX=g++
#CXXFLAGS=-fopenmp -static -O3
CXXFLAGS=-fopenmp -fPIC -pipe -O3
OS = $(shell uname)
SHVER = 1

all: pmf_R.so

pmf_R.so: ../ccd-r1.o ../util.o pmf_R.cpp pmf_R.h
	R CMD SHLIB ${CXXFLAGS} -o pmf_R.so pmf_R.cpp ../util.o ../ccd-r1.o 

clean:
	rm -rf *.o pmf_R.so.* *.so

