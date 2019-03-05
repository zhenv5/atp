make clean && make && ./omp-pmf-train -np 1 -q 1  -k 5 -t 5  toy-example model &&  ./omp-pmf-predict toy-example/test.ratings model output_file
