#!/usr/bin/env Rscript
########################################
## input configurations
########################################
args <- commandArgs(trailingOnly = TRUE)
alpha_ind <- as.integer(args[1])
Gamma_ind <- as.integer(args[2])
seed <- as.integer(args[3])
w_dim <- as.integer(args[4])  # W 的维度（可选，默认 20）
if (is.na(w_dim)) w_dim <- 20  # 如果没有提供，默认为 20

alphas = seq(0.1,0.9,by=0.1)
gammas = c(1.5,2,2.5,3,5)
# coverage target 1-alpha
alpha = alphas[alpha_ind] 
# confounding level Gamma
Gamma = gammas[Gamma_ind] 

########################################
## load libraries
########################################
suppressPackageStartupMessages(library(grf))
options(warn=-1)

########################################
## load util functions
########################################
source("../utils/util_synthetic.R")
source("../utils/util_ate.R")
load("synthetic_population.RData")
cat(paste(" - Running counterfactual prediction on synthetic data with marginally-valid algorithm, alpha", alpha, ", Gamma",Gamma,
          ", w_dim", w_dim, ", seed", seed, "\n"), sep = '')

########################################
## Output directory（按 w_dim 区分）
########################################
base_out_dir <- "/projects/p32685/cfsensitivity_results/realdata_prox/"
out_dir <- file.path(base_out_dir, paste0("wdim_", w_dim))
if(!dir.exists(out_dir)){
  dir.create(out_dir, recursive = TRUE)
}


set.seed(seed)
########################################
## generate synthetic data
########################################
n = 10000
u_dim = 20  # U 固定为 20 维（与 simulation 一致）
data = syn.gen(n, pop.data, Gamma, 1)
# 将 U 从 1 维扩展为 20 维（与 simulation 一致）
# 使用原始 U 作为第一维，其他维度独立生成
set.seed(seed + 1000)  # 固定种子确保可重复
U_original = data$U  # 原始的 1 维 U
U_matrix = matrix(0, nrow = n, ncol = u_dim)
U_matrix[, 1] = U_original
# 其他维度基于原始 U 和独立噪声生成
for (i in 2:u_dim) {
  U_matrix[, i] = U_original * (0.5 + 0.5 * (i-1)/u_dim) + rnorm(n) * sd(U_original) * 0.3
}
data$U = U_matrix  # 替换为 20 维 U

# data splitting
re.ind = sample(n,n)
train.ind = re.ind[1:floor(3*n/10)]
calib.ind = re.ind[floor(3*n/10+1):floor(9*n/10)]
test.ind = re.ind[floor(9*n/10+1):n]

########################################
## fit on the training fold
########################################
train.X = (data$X[train.ind,])[data$T[train.ind]==1,]
train.Y = (data$Y[train.ind])[data$T[train.ind]==1]
# train the nonconformity score function
train.score = conform.score(train.X, train.Y, "cqr", trained_model=NULL, quantile=1-alpha)
t.mdl = train.score$model
# estimate hat{e}(x)
hat.p = mean(data$T)
e.model = regression_forest(data$X[train.ind,], data$T[train.ind], num.threads = 1)

########################################
## proximal: build proxy W and train e(X,W)
########################################
# 根据 w_dim 生成不同维度的 proxy W（与 simulation 的 est 版一致）
noise_level = 0.5
# 当 w_dim > u_dim 时，创建固定的混合矩阵（确保可重复性）
if (w_dim > u_dim) {
  set.seed(seed + 999)  # 固定种子确保混合矩阵可重复
  extra_cols = w_dim - u_dim
  M_extra = matrix(rnorm(u_dim * extra_cols), nrow=u_dim, ncol=extra_cols)
} else {
  M_extra = NULL
}

# 为全部训练数据生成 proxy W（与 simulation 逻辑一致）
if (w_dim < u_dim) {
  W.full = data$U[, 1:w_dim] + matrix(rnorm(n * w_dim) * noise_level, nrow=n, ncol=w_dim)
} else if (w_dim == u_dim) {
  W.full = data$U + matrix(rnorm(n * u_dim) * noise_level, nrow=n, ncol=u_dim)
} else {
  extra_cols = w_dim - u_dim
  W_extra.full = data$U %*% M_extra + matrix(rnorm(n * extra_cols) * noise_level, nrow=n, ncol=extra_cols)
  W.full = cbind(
    data$U + matrix(rnorm(n * u_dim) * noise_level, nrow=n, ncol=u_dim),
    W_extra.full
  )
}
train.W.full = W.full[train.ind, , drop = FALSE]
train.XW.full = cbind(data$X[train.ind,], train.W.full)
e.model.prox = regression_forest(train.XW.full, data$T[train.ind], num.threads = 1)

########################################
# helper for proximal bounds without Gamma
########################################
compute_lx_ux_from_e <- function(e_model, X_mat, W_grid, pbar, eps = 1e-3) {
  n <- nrow(X_mat)
  K <- nrow(W_grid)
  if (n < 1 || K < 1) {
    stop("Empty X_mat or W_grid in compute_lx_ux_from_e")
  }
  X_rep <- X_mat[rep(seq_len(n), each = K), , drop = FALSE]
  W_rep <- do.call(rbind, replicate(n, W_grid, simplify = FALSE))
  XW <- cbind(X_rep, W_rep)
  ex_pred <- predict(e_model, XW)$predictions
  ex_mat <- matrix(ex_pred, nrow = n, byrow = TRUE)
  ex_mat <- pmax(pmin(ex_mat, 1 - eps), eps)
  # 使用真实的 min/max（与 simulation est 版一致）
  emax <- apply(ex_mat, 1, max)
  emin <- apply(ex_mat, 1, min)
  lx <- pbar / emax
  ux <- pbar / emin
  return(list(lx = lx, ux = ux))
}

########################################
## calibration 
########################################
calib.X = (data$X[calib.ind,])[data$T[calib.ind]==1,] 
calib.Y = (data$Y[calib.ind])[data$T[calib.ind]==1]
calib.T = data$T[calib.ind]
calib.ex = predict(e.model, newdata=calib.X)$predictions 
n_calib = length(calib.Y)
# lower and upper bounds of weight function
calib.lx = hat.p * (1 + (1-calib.ex) / (calib.ex*Gamma))
calib.ux = hat.p * (1 + Gamma * (1-calib.ex) / (calib.ex))
calib.nc.w = hat.p / calib.ex
# non-conformity score on calibration data
calib.score = conform.score(calib.X, calib.Y, "cqr", trained_model=t.mdl, quantile=1-alpha)$score
calib.all = data.frame("score" = calib.score, "lx" = calib.lx, "ux" = calib.ux, "wx" = calib.nc.w)
calib.all = calib.all[order(calib.all$score),] 
rownames(calib.all) = 1:dim(calib.all)[1]

########################################
## test fold
########################################
test.X = data$X[test.ind,] 
test.Y1 = data$Y1[test.ind]
test.ex = predict(e.model, newdata=test.X, num.threads=1)$predictions
test.lx = hat.p * (1+ 1/Gamma * (1-test.ex)/test.ex)
test.ux = hat.p * (1+ Gamma* (1-test.ex)/(test.ex))
test.pred = predict(t.mdl, test.X, quantile=c(alpha/2, 1-alpha/2)) 
# ✅ 兼容 grf 新旧版本
if (is.list(test.pred) && "predictions" %in% names(test.pred)) {
  test.pred <- test.pred$predictions
}
n_test = length(test.Y1)

########################################
## proximal: prepare W grid and bounds
########################################
set.seed(seed + 123)
w_grid_size <- min(50, nrow(train.W.full))
W.grid <- train.W.full[sample(nrow(train.W.full), w_grid_size), , drop = FALSE]

prox_bounds_calib <- compute_lx_ux_from_e(e.model.prox, calib.X, W.grid, pbar = hat.p)
calib.lx.prox <- prox_bounds_calib$lx
calib.ux.prox <- prox_bounds_calib$ux

prox_bounds_test <- compute_lx_ux_from_e(e.model.prox, test.X, W.grid, pbar = hat.p)
test.lx.prox <- prox_bounds_test$lx
test.ux.prox <- prox_bounds_test$ux


###################################
# the confounding-aware algorithm 
###################################

cat(" - Computing the robust weighted conformal inference...")

# partial sums for confounding-aware
sum.num = rep(0,n_calib) # for numerator
sum.den = rep(0,n_calib) # for denominator
sum.num[1] = calib.all$lx[1]
sum.den[1] = calib.all$lx[1] + sum(calib.all$ux[2:n_calib])
for (k in 2:n_calib){
  sum.num[k] = sum.num[k-1] + calib.all$lx[k]
  sum.den[k] = sum.den[k-1] - calib.all$ux[k] + calib.all$lx[k] 
}

# confounding-aware prediction 
c.test.lo = rep(0,n_test)
c.test.hi = rep(0,n_test)
for (ii in 1:n_test){
  ratios = sum.num / (sum.den + test.ux[ii])
  kstar = min(which(ratios>1-alpha))
  v.kstar = calib.all$score[kstar]
  c.test.lo[ii] = test.pred[ii,1]-v.kstar
  c.test.hi[ii] = test.pred[ii,2]+v.kstar
}
# evaluate coverage on test data
c.cover = (c.test.lo <= test.Y1) * (c.test.hi >= test.Y1)

cat("Done.\n")

#####################################
# the confounding-unaware algorithm 
#####################################

cat(" - Computing the vanilla weighted conformal inference...")

nc.sum = rep(0,n_calib)
nc.sum[1] = calib.all$wx[1]
for (k in 2:n_calib){
  nc.sum[k] = nc.sum[k-1] + calib.all$wx[k]
}

nc.test.lo = rep(0,n_test)
nc.test.hi = rep(0,n_test)
nc.test.weight = hat.p / test.ex

for (ii in 1:n_test){
  nc.ratios = nc.sum / (nc.sum[n_calib] + nc.test.weight[ii])
  nc.kstar = min(which(nc.ratios>1-alpha))
  nc.v.kstar = calib.all$score[nc.kstar]
  nc.test.lo[ii] = test.pred[ii,1] - nc.v.kstar
  nc.test.hi[ii] = test.pred[ii,2] + nc.v.kstar
}
nc.cover = (nc.test.lo <= test.Y1) * (nc.test.hi >= test.Y1)

cat("Done.\n")

########################################
## proximal CSPCI method (estimated e(X,W))
########################################

cat(" - Computing the proximal conformal inference...")

prox.sum.num = rep(0, n_calib) # for numerator
prox.sum.den = rep(0, n_calib) # for denominator
prox.sum.num[1] = calib.lx.prox[1]
prox.sum.den[1] = calib.lx.prox[1] + sum(calib.ux.prox[2:n_calib])
for (k in 2:n_calib){
  prox.sum.num[k] = prox.sum.num[k-1] + calib.lx.prox[k]
  prox.sum.den[k] = prox.sum.den[k-1] - calib.ux.prox[k] + calib.lx.prox[k]
}

prox.test.lo = rep(0, n_test)
prox.test.hi = rep(0, n_test)
for (ii in 1:n_test){
  prox.ratios = prox.sum.num / (prox.sum.den + test.ux.prox[ii])
  prox.kstar = min(which(prox.ratios > 1 - alpha))
  prox.v.kstar = calib.all$score[prox.kstar]
  prox.test.lo[ii] = test.pred[ii,1] - prox.v.kstar
  prox.test.hi[ii] = test.pred[ii,2] + prox.v.kstar
}
prox.cover = (prox.test.lo <= test.Y1) * (prox.test.hi >= test.Y1)

cat("Done.\n")

########################################
# output summary of test   
########################################
res = data.frame("c.cov" = mean(c.cover), "c.len"=mean(c.test.hi-c.test.lo),
                 "nc.cov" = mean(nc.cover), "nc.len"=mean(nc.test.hi-nc.test.lo),
                 "prox.cov" = mean(prox.cover), "prox.len"=mean(prox.test.hi - prox.test.lo),
                 "n_train" = length(train.ind), "n_calib"=n_calib,
                 "u_dim" = u_dim, "w_dim" = w_dim,
                 "gamma"=Gamma, "alpha" = alpha, "seed"=seed, "method" = "marginal_plus_proximal_est")

write.csv(res, paste(out_dir, "syn_pred_marginal_prox_alpha_",alpha_ind,"_gamma_",Gamma_ind,"_wdim_",w_dim,"_seed_",seed,".csv",sep=''))