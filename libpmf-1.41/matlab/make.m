% This make.m is for MATLAB and OCTAVE under Windows, Mac, and Unix

try
	Type = ver;
	% This part is for OCTAVE
	if(strcmp(Type(1).Name, 'Octave') == 1)
		system('make -C ../ ccd-r1.o dsgd.o als.o util.o');
		mex -lgomp -llapack_atlas -lf77blas -lcblas -latlas -lgfortran pmf.cpp ../ccd-r1.o ../als.o ../dsgd.o ../util.o ../openblas/lib/libopenblas.a
		
	% This part is for MATLAB
	
	% Add -largeArrayDims on 64-bit machines of MATLAB
	else
		system('make -C ../ ccd-r1.o util.o');
		mex -largeArrayDims CFLAGS="\$CFLAGS -fopenmp " LDFLAGS="\$LDFLAGS -fopenmp " COMPFLAGS="\$COMPFLAGS -openmp" -cxx pmf_train.cpp ../ccd-r1.o ../util.o 
	end
catch
	fprintf('If make.m failes, please check README about detailed instructions.\n');
end
