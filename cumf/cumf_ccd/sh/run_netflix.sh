out=inner.csv
# for lambda in .058
# do
# 	for in in 2 5 
# 	do
# 		echo "hoitese"
# 		#./omp-pmf-train -a 100000 -b 60000 -k $t -l $lambda -t 20 ../dataset/Yahoo_music ../dataset/Yahoo_music/test2.txt model  >> $out
# 		./omp-pmf-train -a 100000 -b 20000 -l .058 -T $in -k 40 -t 20  ../dataset/Netflix ../dataset/Netflix/netflix_mme_wo_header model &>> $out
# 	done
# done

out1=inner.csv
for lambda in 1.2
do
	for in in 2 5
	do
		echo "hoitese"
		#./omp-pmf-train -a 100000 -b 60000 -k $t -l $lambda -t 20 ../dataset/Yahoo_music ../dataset/Yahoo_music/test2.txt model  >> $out
		./omp-pmf-train -a 100000 -b 100000 -l $lambda -T $in -k 40 -t 20  ../dataset/Yahoo_music ../dataset/Yahoo_music/test.txt model &>> $out1
	done
done




# for t in 10 20 40 80 
# do
#     echo "tile, 7000, 10000 $t " 
# 	./omp-pmf-train -a 7000 -b 10000 -l .058 -k $t -t 10  ../dataset/Netflix ../dataset/Netflix/netflix_mme_wo_header model >> $out
# done
# for t in 10 20 40 80 
# do
#     echo "tile 10k 10k $t " 
# 	./omp-pmf-train -a 10000 -b 10000 -l .058 -k $t -t 10  ../dataset/Netflix ../dataset/Netflix/netflix_mme_wo_header model >> $out
# done


# out_nv=nvprof_netflix_binned_fused_10k-10k_Tiled_sliced.csv
# metric1=l1_cache_global_hit_rate,tex_cache_hit_rate,l2_l1_read_hit_rate,l2_texture_read_hit_rate
# metric2=dram_read_transactions,dram_write_transactions,achieved_occupancy
# echo "nvprof NetflixbinnedfusedsTILE"
# nvprof --print-gpu-trace --metrics $metric1,$metric2 --csv ./omp-pmf-train -a 10000 -b 10000 -l .058 -k 40 -t 1 ../dataset/Netflix ../dataset/Netflix/netflix_mme_wo_header model &> $out_nv
# out_nv1=nvprof_netflix_binned_fused_5k-20k_Tiled.csv
# echo "nvprof NetflixbinnedfusedsTILE"
# nvprof --print-gpu-trace --metrics $metric1,$metric2 --csv ./omp-pmf-train -a 5000 -b 20000 -l $lambda -k 40 -t 1 ../dataset/Netflix ../dataset/Netflix/netflix_mme_wo_header model &> $out_nv1

# tile_size=10000
# echo $tile_size
# nvprof --print-gpu-trace --metrics $metric1,$metric2 --csv ./omp-pmf-train -a $tile_size -l .05 -q 1 -k 40 -t 1 ../dataset/Netflix ../dataset/Netflix/netflix_mme_wo_header model &> $tile_size"_Binned_k40_nvprof".csv



# tile_size=20000
# echo $tile_size
# nvprof --print-gpu-trace --metrics $metric1,$metric2 --events $event1,$event2 --csv ./omp-pmf-train -a $tile_size -l .05 -q 1 -k 100 -t 1 ../dataset/Netflix model &> $tile_size"_k_100_nvprof".csv
