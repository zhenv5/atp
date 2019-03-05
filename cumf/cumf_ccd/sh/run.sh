out=netflix_rmse_100
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15;
do
	echo "iter $i" >> $out
	# ./omp-pmf-train -l .05 -k 40 -t $i ../dataset/movielens model >> $out
	# ./omp-pmf-predict ../dataset/movielens/test.txt model output_file >> $out
	./omp-pmf-train -l .05 -k 100 -t $i ../dataset/Netflix model
	./omp-pmf-predict ../dataset/Netflix/netflix_mme_wo_header model output_file >> $out
done

#make clean && make && ./omp-pmf-train -l .05 -q 1 -k 10 -t 1  ../dataset/Netflix model && ./omp-pmf-predict ../dataset/Netflix/netflix_mme_wo_header model output_file
