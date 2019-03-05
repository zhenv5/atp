
library(reshape2);

load.library = function() {
	srcpath =  normalizePath((function() { attr(body(sys.function()), "srcfile") })()$filename);
	srcdir = dirname(srcpath)
	dynpath = file.path(srcdir, 'pmf_R.so');
	if (!file.exists(dynpath)) {
		cmd = paste('make -C ', srcdir)
		system(cmd)
	} 	
	dyn.load(dynpath);
}

load.library();

pmf.train.coo = function(row.idx = NULL, col.idx = NULL, obs.val = NULL, obs.weight = NULL, param.str=''){
	if (is.null(row.idx) | is.null(col.idx) | is.null(obs.val)) {
		cat("model = pmf.train.coo(row.idx, col.idx, obs.val, obs.weight=NULL, param.str='')\n");
		.Call("print_training_option");
		return (0)
	} else {
		row.idx = as.integer(row.idx);
		col.idx = as.integer(col.idx);
		obs.val = as.numeric(obs.val);
		param.str = as.character(param.str);
		m = as.integer(max(row.idx));
		n = as.integer(max(col.idx));
		row.idx = as.integer(row.idx - 1);
		col.idx = as.integer(col.idx - 1);
		nnz = length(obs.val);
		k = .Call("get_rank_from_param", param.str);
		ret = list();
		ret$W = matrix(0.0, nrow = m, ncol = k);
		ret$H = matrix(0.0, nrow = n, ncol = k);

		obs.weight = as.numeric(obs.weight)
		if (length(obs.weight) == 0) { # no obs.weight
			.Call("pmf_train", m, n, nnz, row.idx, col.idx, obs.val, param.str, ret$W, ret$H);
		} else if (length(obs.weight) == nnz) {
			.Call("pmf_weighted_train", m, n, nnz, row.idx, col.idx, obs.val, obs.weight, param.str, ret$W, ret$H);
		} else {
			cat("length(obs.val) != length(obs.weight)")
			.Call("print_training_option");
			return (0);
		}
		return(ret);
	}
}

pmf.train.matrix = function(mat = NULL, param.str = '', zero_as_missing = TRUE) {
	if (is.null(mat)) {
		cat("model = pmf.train.matrix(mat, param.str='', zero_as_missing = TRUE)\n");
		.Call("print_training_option");
		return (0)
	} else  {
		melten.mat = melt(mat, varnames = c('row.idx', 'col.idx'), value.name = 'val');
		if (zero_as_missing == TRUE) {
			melten.mat = melten.mat[(is.finite(melten.mat$val)) & (melten.mat$val != 0),];
		} else {
			melten.mat = melten.mat[(is.finite(melten.mat$val)),];
		}
		return (pmf.train.coo(row.idx=melten.mat$row.idx, col.idx=melten.mat$col.idx, obs.val=melten.mat$val, param.str=param.str));
	}
}

rm(load.library)
