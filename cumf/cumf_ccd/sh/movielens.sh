make && ./omp-pmf-train -l .05 -k 40 -t 15 ../dataset/movielens model && ./omp-pmf-predict ../dataset/movielens/test.txt model output_file
