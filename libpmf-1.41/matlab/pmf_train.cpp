

#include "mex.h"
#include "../util.h"
#include "../pmf.h"
#include <omp.h>
#include <cstring>

#ifdef MX_API_VER
#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif
#endif

#define CMD_LEN 2048
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL


void exit_with_help()
{
	mexPrintf(
	"Usage: [W H] = pmf_train(R, W, H [, 'pmf_options'])\n"
	"       [W H] = pmf_train(R, [, 'pmf_options'])\n"
	"     R is an m-by-n sparse double matrix\n"
	"     W is an m-by-k dense double matrix\n"
	"     H is an n-by-k dense double matrix\n"
	"     If W and H are given, they will be treated as the initial values,\n"
	"     and \"rank\" will equal to size(W,2).\n"
	"options:\n"
	"    -s type : set type of solver (default 0)\n"    
	"    	 0 -- CCDR1 with fundec stopping condition\n"    
	"    -k rank : set the rank (default 10)\n"    
	"    -n threads : set the number of threads (default 4)\n"    
	"    -l lambda : set the regularization parameter lambda (default 0.1)\n"    
	"    -t max_iter: set the number of iterations (default 5)\n"    
	"    -T max_iter: set the number of inner iterations used in CCDR1 (default 5)\n"    
	"    -e epsilon : set inner termination criterion epsilon of CCDR1 (default 1e-3)\n"     
	"    -q verbose: show information or not (default 0)\n"
	"    -N do_nmf: do nmf (default 0)\n"
	//"    -p do_predict: do prediction or not (default 0)\n"    
	);
}

// nrhs == 1 or 2 => pmf(R [, 'pmf_options']);
// nrhs == 3 or 4 => pmf(R, W, H [, 'pmf_options']);
parameter parse_command_line(int nrhs, const mxArray *prhs[])
{
	parameter param;   // default values have been set by the constructor 
	int i, argc = 1;
	int option_pos = -1;
	char cmd[CMD_LEN];
	char *argv[CMD_LEN/2];

	if(nrhs < 1)
		return param;

	// put options in argv[]
	if(nrhs == 2) option_pos = 1;
	if(nrhs == 4) option_pos = 3;
	if(option_pos>0)
	{
		mxGetString(prhs[option_pos], cmd,  mxGetN(prhs[option_pos]) + 1);
		if((argv[argc] = strtok(cmd, " ")) != NULL)
			while((argv[++argc] = strtok(NULL, " ")) != NULL)
				;
	}

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 's':
				param.solver_type = atoi(argv[i]);
				break;

			case 'k':
				param.k = atoi(argv[i]);
				break;

			case 'n':
				param.threads = atoi(argv[i]);
				break;

			case 'l':
				param.lambda = atof(argv[i]);
				break;

			case 't':
				param.maxiter = atoi(argv[i]);
				break;

			case 'T':
				param.maxinneriter = atoi(argv[i]);
				break;

			case 'e':
				param.eps = atof(argv[i]);
				param.eta0 = atof(argv[i]);
				break;

			case 'B':
				param.num_blocks = atoi(argv[i]);
				break;

			case 'm':
				param.lrate_method = atoi(argv[i]);
				break;

			case 'u':
				param.betaup = atof(argv[i]);
				break;

			case 'd':
				param.betadown = atof(argv[i]);
				break;

			case 'p':
				param.do_predict = atoi(argv[i]);
				break;

			case 'q':
				param.verbose = atoi(argv[i]);
				break;

			case 'N':
				param.do_nmf = atoi(argv[i]) == 1? true : false;
				break;

			default:
				mexPrintf("unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	if (param.do_predict!=0) 
		param.verbose = 1;

	if (nrhs > 2) {
		if(mxGetN(prhs[1]) != mxGetN(prhs[2])) 
			mexPrintf("Dimensions of W and H do not match!\n");
		param.k = (int)mxGetN(prhs[1]);
		mexPrintf("Change param.k to %d.\n", param.k);
	}

	return param;
}

static void fake_answer(mxArray *plhs[])
{
	plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

int transpose(const mxArray *M, mxArray **Mt) {
	mxArray *prhs[1] = {const_cast<mxArray *>(M)}, *plhs[1];
	if(mexCallMATLAB(1, plhs, 1, prhs, "transpose"))
	{
		mexPrintf("Error: cannot transpose training instance matrix\n");
		return -1;
	}
	*Mt = plhs[0];
	return 0;
}

// convert matlab sparse matrix to C smat fmt

class mxSparse_iterator_t: public entry_iterator_t {
	private:
		mxArray *Mt;
		mwIndex *ir_t, *jc_t;
		double *v_t;
		size_t	rows, cols, cur_idx, cur_row;
	public:
		mxSparse_iterator_t(const mxArray *M){
			rows = mxGetM(M); cols = mxGetN(M);
			nnz = *(mxGetJc(M) + cols); 
			transpose(M, &Mt);
			ir_t = mxGetIr(Mt); jc_t = mxGetJc(Mt); v_t = mxGetPr(Mt);
			cur_idx = cur_row = 0;
		}
		rate_t next() {
			int i = 1, j = 1;
			double v = 0;
			while (cur_idx >= jc_t[cur_row+1]) ++cur_row;
			if (nnz > 0) --nnz;
			else fprintf(stderr,"Error: no more entry to iterate !!\n");
			rate_t ret(cur_row, ir_t[cur_idx], v_t[cur_idx]);
			cur_idx++;
			return ret;
		}
		~mxSparse_iterator_t(){
			mxDestroyArray(Mt);
		}

};
smat_t mxSparse_to_smat(const mxArray *M, smat_t &R) {
	long rows = mxGetM(M), cols = mxGetN(M), nnz = *(mxGetJc(M) + cols); 
	mxSparse_iterator_t entry_it(M);
	R.load_from_iterator(rows, cols, nnz, &entry_it);
	return R;
}

// convert matab dense matrix to column fmt
int mxDense_to_matCol(const mxArray *mxM, mat_t &M) {
	unsigned long rows = mxGetM(mxM), cols = mxGetN(mxM);
	double *val = mxGetPr(mxM);
	M = mat_t(cols, vec_t(rows,0));
	for(unsigned long c = 0, idx = 0; c < cols; ++c) 
		for(unsigned long r = 0; r < rows; ++r)
			M[c][r] = val[idx++];
	return 0;
}

int matCol_to_mxDense(const mat_t &M, mxArray *mxM) {
	unsigned long cols = M.size(), rows = M[0].size();
	double *val = mxGetPr(mxM);
	if(cols != mxGetN(mxM) || rows != mxGetM(mxM)) {
		mexPrintf("matCol_to_mxDense fails (dimensions do not match)\n");
		return -1;
	}

	for(unsigned long c = 0, idx = 0; c < cols; ++c)
		for(unsigned long r = 0; r < rows; r++) 
			val[idx++] = M[c][r];
	return 0;
}

// convert matab dense matrix to row fmt
int mxDense_to_matRow(const mxArray *mxM, mat_t &M) {
	unsigned long rows = mxGetM(mxM), cols = mxGetN(mxM);
	double *val = mxGetPr(mxM);
	M = mat_t(rows, vec_t(cols,0));
	for(unsigned long c = 0, idx = 0; c < cols; ++c) 
		for(unsigned long r = 0; r < rows; ++r)
			M[r][c] = val[idx++];
	return 0;
}

int matRow_to_mxDense(const mat_t &M, mxArray *mxM) {
	unsigned long rows = M.size(), cols = M[0].size();
	double *val = mxGetPr(mxM);
	if(cols != mxGetN(mxM) || rows != mxGetM(mxM)) {
		mexPrintf("matRow_to_mxDense fails (dimensions do not match)\n");
		return -1;
	}

	for(unsigned long c = 0, idx = 0; c < cols; ++c)
		for(unsigned long r = 0; r < rows; r++) 
			val[idx++] = M[r][c];
	return 0;
}

int run_ccdr1(mxArray *plhs[], int nrhs, const mxArray *prhs[], parameter &param){
	mxArray *mxR, *mxRt;
	mxArray *mxW, *mxH;
	smat_t R;
	mat_t W, H;
	testset_t T;  // Dummy

	//mxR = mxDuplicateArray(prhs[0]);
	//transpose(mxR, &mxRt);
	mxSparse_to_smat(prhs[0], R);

	// fix random seed to have same results for each run
	// (for random initialization)
	srand(1);
	srand48(0L);

	// Initialization of W and H
	if(nrhs > 2) { 
		mxDense_to_matCol(prhs[1], W);
		mxDense_to_matCol(prhs[2], H);
	} else {
		initial_col(W, param.k, R.rows);
		initial_col(H, param.k, R.cols);
		mexPrintf("W norm %g H norm %g\n", norm(W), norm(H));
	}

	// Execute the program
	double time = omp_get_wtime();
	ccdr1(R, W, H, T, param);
	printf("Wall-time: %lg secs\n", omp_get_wtime() - time);

	// Write back the result
	plhs[0] = mxW = mxCreateDoubleMatrix(R.rows, param.k, mxREAL);
	plhs[1] = mxH = mxCreateDoubleMatrix(R.cols, param.k, mxREAL);
	matCol_to_mxDense(W, mxW);
	matCol_to_mxDense(H, mxH);

	// Destroy matrix we allocated in this function
	mxDestroyArray(mxR);
	mxDestroyArray(mxRt);
	return 0;
}

// Interface function of matlab
// now assume prhs[0]: A, prhs[1]: W, prhs[0]
void mexFunction( int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[] )
{
	parameter param;
	// fix random seed to have same results for each run
	// (for cross validation)
	srand(1);

	// Transform the input Matrix to libsvm format
	if(nrhs > 0 && nrhs < 5)
	{
		if(!mxIsDouble(prhs[0]) || !mxIsSparse(prhs[0])) {
			mexPrintf("Error: matrix must be double and sparse\n");
			fake_answer(plhs);
			return;
		}

		param = parse_command_line(nrhs, prhs);

		switch (param.solver_type){
			case CCDR1:
				run_ccdr1(plhs, nrhs, prhs, param);
				break;
			default:
				fprintf(stderr, "Error: wrong solver type (%d)!\n", param.solver_type);
				exit_with_help();
				fake_answer(plhs);
				break;
		}
	} else {
		exit_with_help();
		fake_answer(plhs);
	}

}


