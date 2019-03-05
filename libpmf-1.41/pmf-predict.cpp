#include "util.h"
#include "pmf.h"
#include <cstring>


void exit_with_help()
{
	printf(
	"Usage: omp-pmf-predict test_file model output_file\n"
	);
	exit(1);
}


int main(int argc, char* argv[]){
	if(argc != 4) 
		exit_with_help();	

	char *test_file = argv[1];
	char *model_file = argv[2];
	char *output_file = argv[3];
	FILE *test_fp = NULL, *model_fp=NULL, *output_fp=NULL;

	if(test_file) {
		test_fp = fopen(test_file, "r");
		if(test_fp == NULL)
		{
			fprintf(stderr,"can't open test file %s\n",test_file);
			exit(1);
		}
	}
	if(output_file) {
		output_fp = fopen(output_file, "wb");
		if(output_fp == NULL)
		{
			fprintf(stderr,"can't open output file %s\n",output_file);
			exit(1);
		}
	}
	if(model_file) {
		model_fp = fopen(model_file, "rb");
		if(model_fp == NULL)
		{
			fprintf(stderr,"can't open model file %s\n",model_file);
			exit(1);
		}
	}

	mat_t W = load_mat_t(model_fp, true);
	mat_t H = load_mat_t(model_fp, true);

	int rank = W[0].size();

	int i, j;
	double v, rmse = 0;
	size_t num_insts = 0;
	while(fscanf(test_fp, "%d %d %lf", &i, &j, &v) != EOF) {
		double pred_v = 0;
#pragma omp parallel for  reduction(+:pred_v)
		for(int t = 0; t < rank; t++)
			pred_v += W[i-1][t] * H[j-1][t];
		num_insts ++;
		rmse += (pred_v - v)*(pred_v - v);
		fprintf(output_fp, "%lf\n", pred_v);
	}
	rmse = sqrt(rmse/num_insts);
	printf("test RMSE = %g\n", rmse);

	return 0;
}

