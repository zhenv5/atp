CXX=g++-7
#CXXFLAGS=-fopenmp -static -O3
CXXFLAGS=-fopenmp -fPIC -pipe -O3
OS = $(shell uname)
SHVER = 1

all: lib

lib: ../ccd-r1.o ../util.o pmf_py.o
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-dynamiclib -Wl,-install_name,pmf_py.so.$(SHVER)"; \
	else \
		SHARED_LIB_FLAG="-shared -Wl,-soname,pmf_py.so.$(SHVER)"; \
	fi; \
	$(CXX) ${CXXFLAGS} $${SHARED_LIB_FLAG} pmf_py.o ../util.o ../ccd-r1.o -o pmf_py.so.$(SHVER)

pmf_py.o: pmf_py.cpp pmf_py.h
	${CXX} ${CXXFLAGS} -c -o pmf_py.o pmf_py.cpp 

clean:
	rm -rf *.o pmf_py.so.* *pyc

