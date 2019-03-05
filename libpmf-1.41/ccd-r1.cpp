#include "util.h"
#include "pmf.h"
#define kind dynamic,500

// CCD rank-one 

inline double RankOneUpdate(const smat_t &R, const int j, const vec_t &u, const double lambda, const double vj, double *redvar, int do_nmf){
	double g=0, h=lambda;
	if(R.col_ptr[j+1]==R.col_ptr[j]) return 0;
	for(long idx=R.col_ptr[j]; idx < R.col_ptr[j+1]; ++idx) {
		int i = R.row_idx[idx];
		g += u[i]*R.val[idx]; 
		h += u[i]*u[i];
	}
	double newvj = g/h, tmp = 0, delta = 0, fundec = 0;
	if(do_nmf>0 & newvj < 0) {
		newvj = 0;
		delta = vj; // old - new
		fundec = -2*g*vj; + h*vj*vj;
	} else {
		delta = vj - newvj;
		fundec = h*delta*delta;
	}
	//double delta = vj - newvj;
	//double fundec = h*delta*delta;
	//double lossdec = fundec - lambda*delta*(vj+newvj);
	//double gnorm = (g-h*vj)*(g-h*vj); 
	*redvar += fundec;
	//*redvar += lossdec;
	return newvj;
}

inline double UpdateRating(smat_t &R, const vec_t &Wt, const vec_t &Ht, const vec_t &oldWt, const vec_t &oldHt) {
	double loss=0;
#pragma omp parallel for  schedule(kind) reduction(+:loss)
		for(int c =0; c < R.cols; ++c){
			double Htc = Ht[c], oldHtc = oldHt[c], loss_inner = 0;
			for(long idx=R.col_ptr[c]; idx < R.col_ptr[c+1]; ++idx){
				R.val[idx] -=  Wt[R.row_idx[idx]]*Htc-oldWt[R.row_idx[idx]]*oldHtc;
				loss_inner += R.val[idx]*R.val[idx];
			}
			loss += loss_inner;
		}
	return loss;	
}
inline double UpdateRating(smat_t &R, const vec_t &Wt2, const vec_t &Ht2) {
	double loss=0;
#pragma omp parallel for schedule(kind) reduction(+:loss)
		for(int c =0; c < R.cols; ++c){
			double Htc = Ht2[2*c], oldHtc = Ht2[2*c+1], loss_inner = 0;
			for(long idx=R.col_ptr[c]; idx < R.col_ptr[c+1]; ++idx){
				R.val[idx] -=  Wt2[2*R.row_idx[idx]]*Htc-Wt2[2*R.row_idx[idx]+1]*oldHtc;
				loss_inner += R.val[idx]*R.val[idx];
			}
			loss += loss_inner;
		}
	return loss;	
}
inline double UpdateRating(smat_t &R, const vec_t &Wt, const vec_t &Ht, bool add) {
	double loss=0;
	if(add) {
#pragma omp parallel for schedule(kind) reduction(+:loss)
		for(int c =0; c < R.cols; ++c){
			double Htc = Ht[c], loss_inner = 0;
			for(long idx=R.col_ptr[c]; idx < R.col_ptr[c+1]; ++idx){
				R.val[idx] +=  Wt[R.row_idx[idx]]*Htc;
				loss_inner += (R.with_weights? R.weight[idx]: 1.0)*R.val[idx]*R.val[idx];
			}
			loss += loss_inner;
		}
		return loss;	
	} else {
#pragma omp parallel for schedule(kind) reduction(+:loss)
		for(int c =0; c < R.cols; ++c){
			double Htc = Ht[c], loss_inner = 0;
			for(long idx=R.col_ptr[c]; idx < R.col_ptr[c+1]; ++idx){
				R.val[idx] -=  Wt[R.row_idx[idx]]*Htc;
				loss_inner += (R.with_weights? R.weight[idx]: 1.0)*R.val[idx]*R.val[idx];
			}
			loss += loss_inner;
		}
		return loss;	
	}
}

// Cyclic Coordinate Descent for Matrix Factorization
void ccdr1(smat_t &R, mat_t &W, mat_t &H, testset_t &T, parameter &param){
	int k = param.k;
	int maxiter = param.maxiter;
	int inneriter = param.maxinneriter;
	int num_threads_old = omp_get_num_threads();
	double lambda = param.lambda;
	double eps = param.eps;
	double Itime = 0, Wtime = 0, Htime = 0, Rtime = 0, start = 0, oldobj=0;
	long num_updates = 0;
	double reg=0,loss;

	omp_set_num_threads(param.threads);

	// Create transpose view of R
	smat_t Rt;
	Rt = R.transpose();
	// initial value of the regularization term
	// H is a zero matrix now.
	for(int t=0;t<k;++t) for(long c=0;c<R.cols;++c) H[t][c] = 0; 
	for(int t=0;t<k;++t) for(long r=0;r<R.rows;++r) reg += W[t][r]*W[t][r]*R.nnz_of_row(r);
	
	vec_t oldWt(R.rows), oldHt(R.cols);
	vec_t u(R.rows), v(R.cols);
	for(int oiter = 1; oiter <= maxiter; ++oiter) {
		double gnorm = 0, initgnorm=0;
		double rankfundec = 0;
		double fundec_max = 0;
		int early_stop = 0;
		for(int tt=0; tt < k; ++tt) {
			int t = tt;
			if(early_stop >= 5) break;
			//if(oiter>1) { t = rand()%k; }
			start = omp_get_wtime();
			vec_t &Wt = W[t], &Ht = H[t];
#pragma omp parallel for
			for(int i = 0; i < R.rows; ++i) oldWt[i] = u[i]= Wt[i];
#pragma omp parallel for
			for(int i = 0; i < R.cols; ++i) {v[i]= Ht[i]; oldHt[i] = (oiter == 1)? 0: v[i];}

			// Create Rhat = R - Wt Ht^T
			if (oiter > 1) {
				UpdateRating(R, Wt, Ht, true);
				UpdateRating(Rt, Ht, Wt, true);
			} 
			Itime += omp_get_wtime() - start;

			gnorm = 0, initgnorm=0;
			double innerfundec_cur = 0, innerfundec_max = 0;
			int maxit = inneriter; 	
			//	if(oiter > 1) maxit *= 2;
			for(int iter = 1; iter <= maxit; ++iter){
				// Update H[t]
				start = omp_get_wtime();
				gnorm = 0;
				innerfundec_cur = 0;
#pragma omp parallel for schedule(kind) shared(u,v) reduction(+:innerfundec_cur)
				for(long c = 0; c < R.cols; ++c)
					v[c] = RankOneUpdate(R, c, u, lambda*(R.col_ptr[c+1]-R.col_ptr[c]), v[c], &innerfundec_cur, param.do_nmf);
				num_updates += R.cols;
				Htime += omp_get_wtime() - start;
				// Update W[t]
				start = omp_get_wtime();
#pragma omp parallel for schedule(kind) shared(u,v) reduction(+:innerfundec_cur)
				for(long c = 0; c < Rt.cols; ++c)
					u[c] = RankOneUpdate(Rt, c, v, lambda*(Rt.col_ptr[c+1]-Rt.col_ptr[c]), u[c], &innerfundec_cur, param.do_nmf);
				num_updates += Rt.cols;
				if((innerfundec_cur < fundec_max*eps))  {
					if(iter==1) early_stop+=1;
					break; 
				}
				rankfundec += innerfundec_cur;
				innerfundec_max = max(innerfundec_max, innerfundec_cur);
				// the fundec of the first inner iter of the first rank of the first outer iteration could be too large!!
				if(!(oiter==1 && t == 0 && iter==1))
					fundec_max = max(fundec_max, innerfundec_cur);
				Wtime += omp_get_wtime() - start;
			}

			// Update R and Rt
			start = omp_get_wtime();
#pragma omp parallel for
			for(int i = 0; i < R.rows; ++i) Wt[i]= u[i];
#pragma omp parallel for
			for(int i = 0; i < R.cols; ++i) Ht[i]= v[i];
			loss = UpdateRating(R, u, v, false);
			loss = UpdateRating(Rt, v, u, false);
			Rtime += omp_get_wtime() - start;

			for(long c = 0; c < R.cols; ++c) {
				reg += R.nnz_of_col(c)*Ht[c]*Ht[c];
				reg -= R.nnz_of_col(c)*oldHt[c]*oldHt[c];
			}
			for(long c = 0; c < Rt.cols; ++c) {
				reg += Rt.nnz_of_col(c)*(Wt[c]*Wt[c]);
				reg -= Rt.nnz_of_col(c)*(oldWt[c]*oldWt[c]);
			}
			double obj = loss+reg*lambda;
			if(param.verbose)
				printf("iter %d rank %d time %.10g loss %.10g obj %.10g diff %.10g gnorm %.6g reg %.7g ",
						oiter,t+1, Htime+Wtime+Rtime, loss, obj, oldobj - obj, initgnorm, reg);
			oldobj = obj;
			if(T.nnz!=0 and param.do_predict){ 
				if(param.verbose)
					printf("rmse %.10g", calrmse_r1(T, Wt, Ht, oldWt, oldHt)); 
			}
			if(param.verbose) puts("");
			fflush(stdout);
		}
	}
	omp_set_num_threads(num_threads_old);
}

