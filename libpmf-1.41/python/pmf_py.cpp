#include <cstring>
#include "pmf_py.h"

const char* training_option() {
	return 
	"options:\n"
	"    -s type : set type of solver (default 0)\n"    
	"    	 0 -- CCDR1 with fundec stopping condition\n"    
	"    -k rank : set the rank (default 10)\n"    
	"    -n threads : set the number of threads (default 4)\n"    
	"    -l lambda : set the regularization parameter lambda (default 0.1)\n"    
	"    -t max_iter: set the number of iterations (default 5)\n"    
	"    -T max_iter: set the number of inner iterations used in CCDR1 (default 5)\n"    
	"    -e epsilon : set inner termination criterion epsilon of CCDR1 (default 1e-3)\n"     
	"    -p do_predict: do prediction or not (default 0)\n"    
	"    -q verbose: show information or not (default 0)\n"
	"    -N do_nmf: do nmf (default 0)\n";
}


parameter parse_training_command_line(const char *input) {//{{{
	parameter param;   // default values have been set by the constructor 
	char buf[1000];
	char *idx, *val;
	strcpy(buf, input);
	idx = strtok(buf, " \t");
	while (idx!=NULL) {
		val = strtok(NULL, " \t");
		if(val == NULL) {
			fprintf(stderr, "input error idx = %s, buf %s, input %s\n", idx, buf, input);
			break;
		}
		switch (idx[1]) {
			case 's':
				param.solver_type = atoi(val);
				break;

			case 'k':
				param.k = atoi(val);
				break;

			case 'n':
				param.threads = atoi(val);
				break;

			case 'l':
				param.lambda = atof(val);
				break;

			case 'r':
				param.rho = atof(val);
				break;

			case 't':
				param.maxiter = atoi(val);
				break;

			case 'T':
				param.maxinneriter = atoi(val);
				break;

			case 'e':
				param.eps = atof(val);
				param.eta0 = atof(val);
				break;

			case 'B':
				param.num_blocks = atoi(val);
				break;

			case 'm':
				param.lrate_method = atoi(val);
				break;

			case 'u':
				param.betaup = atof(val);
				break;

			case 'd':
				param.betadown = atof(val);
				break;

			case 'p':
				param.do_predict = atoi(val);
				break;

			case 'q':
				param.verbose = atoi(val);
				break;

			case 'N':
				param.do_nmf = atoi(val);
				break;

			default:
				fprintf(stderr,"unknown option: -%c\n", idx[1]);
				break;
		}
		idx = strtok(NULL, " \t");
	}
	if (param.do_predict!=0) 
		param.verbose = 1;
	return param;
}//}}}

void print_tmp(double *W, int m, int n) {
	for(int i = 0; i < m; i++) {
		for(int j = 0; j < n; j++)
			printf("%g ", W[i*n+j]);
		puts("");
	}
}

class coo_iterator_t: public entry_iterator_t{
	private:
		int *row, *col;
		double *val, *weight;
		int idx;
	public:
		coo_iterator_t():idx(0){nnz = 0,with_weights = false;}
		coo_iterator_t(size_t _nnz, int *_row, int *_col, double *_val, double *_weight=NULL) {
			row = _row; col = _col; val = _val; weight = _weight;
			nnz = _nnz;
			idx = 0;
			if(_weight!=NULL) {
				with_weights = true;
			}
		}
		size_t size() {return nnz;}
		rate_t next() {
			if (nnz > 0) {
				--nnz;
			} else {
				fprintf(stderr,"Error: no more entry to iterate !!\n");
			}
			rate_t r(row[idx], col[idx], val[idx], (with_weights? weight[idx] : 1.0));
			idx++;
			return r;
		}
		~coo_iterator_t(){}
};


// ret_W is a m*k matrix stored in column majored order.
// ret_H is an n*k matrix stored in column majored order.
void pmf_train(int m, int n, long nnz, int* i, int *j, double *v, parameter *param, double *ret_W, double *ret_H) {
	smat_t R;
	mat_t W,H;
	testset_t T;
	coo_iterator_t entry_it(nnz, i, j,  v);
	R.load_from_iterator(m, n,  nnz, &entry_it);

	// W, H  here are k*m, k*n
	initial_col(W, param->k, R.rows);
	initial_col(H, param->k, R.cols);

	puts("starts!");
	printf("m %ld n %ld k %d nnz %ld\n", R.rows, R.cols, param->k, R.nnz);
	double time = omp_get_wtime();
	ccdr1(R, W, H, T, *param);
	printf("Wall-time: %lg secs\n", omp_get_wtime() - time);

	for(int t = 0; t < param->k; t++) {
		for(int i = 0; i < m; i++) 
			ret_W[t * m + i] = W[t][i];
		for(int j = 0; j < n; j++)
			ret_H[t * n + j] = H[t][j];
	}
	return;
}

void pmf_weighted_train(int m, int n, long nnz, int* i, int *j, double *v, double *weight, parameter *param, double *ret_W, double *ret_H) {
	smat_t R;
	mat_t W,H;
	testset_t T;
	coo_iterator_t entry_it(nnz, i, j,  v, weight);
	R.load_from_iterator(m, n,  nnz, &entry_it);

	// W, H  here are k*m, k*n
	initial_col(W, param->k, R.rows);
	initial_col(H, param->k, R.cols);

	puts("starts!");
	printf("m %ld n %ld k %d nnz %ld\n", R.rows, R.cols, param->k, R.nnz);
	double time = omp_get_wtime();
	ccdr1(R, W, H, T, *param);
	printf("Wall-time: %lg secs\n", omp_get_wtime() - time);

	for(int t = 0; t < param->k; t++) {
		for(int i = 0; i < m; i++) 
			ret_W[t * m + i] = W[t][i];
		for(int j = 0; j < n; j++)
			ret_H[t * n + j] = H[t][j];
	}
	return;
}


