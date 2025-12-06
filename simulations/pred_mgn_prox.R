#!/usr/bin/env Rscript
########################################
## input configurations
########################################
args <- commandArgs(trailingOnly = TRUE)
p <- as.integer(args[1])
n <- as.integer(args[2])
w_dim <- as.integer(args[3])     # W 的维度
alpha_ind <- as.integer(args[4])
Gamma_ind <- as.integer(args[5])
seed <- as.integer(args[6])

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
source("../utils/util_ate_multi.R")
cat(paste(" - Running pred_mgn_prox: robust + unaware + proximal methods, alpha", alpha, 
          ", Gamma", Gamma, ", n", n, ", p", p, ", w_dim", w_dim, ", seed", seed, "\n"), sep = '')

########################################
## Output directory（真实边界版，按 w_dim 区分）
########################################
base_out_dir <- "/projects/p32685/cfsensitivity_results/simulation_prox_true/"
out_dir <- file.path(base_out_dir, paste0("wdim_", w_dim))
if(!dir.exists(out_dir)){
  dir.create(out_dir, recursive = TRUE)
}

########################################
## Parameter
########################################
alpha0 = 0
n_test = 500
u_dim = 20  # U 固定为 20 维
beta = matrix(c(-0.531,0.126,-0.312,0.018,rep(0,p-4)), nrow=p)
# generate true probability of treatment
pp = mean(data.gen.ate(n*1000,p,Gamma,beta,alpha0,obs=FALSE,u_dim)$T)


set.seed(seed)
########################################
## fit on the training fold
## 【新增】：生成带 proxy W 的数据（U 是 20 维矩阵）
########################################
train.data = data.gen.ate(n,p,Gamma,beta,alpha0,obs=TRUE,u_dim)
train.X = (train.data$X[train.data$T==1,])[1:n,]
train.Y = (train.data$Y1[train.data$T==1])[1:n]
train.U = (train.data$U[train.data$T==1,])[1:n,]  # U 现在是 n x 20 的矩阵

# 【Proximal新增】：根据 w_dim 生成不同维度的 proxy W（仅用于后续采样 W.grid）
noise_level = 0.5
# 当 w_dim > u_dim 时，创建固定的混合矩阵（确保可重复性）
if (w_dim > u_dim) {
  set.seed(seed + 999)  # 固定种子确保混合矩阵可重复
  extra_cols = w_dim - u_dim
  M_extra = matrix(rnorm(u_dim * extra_cols), nrow=u_dim, ncol=extra_cols)
} else {
  M_extra = NULL
}

# train the nonconformity score function（原有方法）
train.score = conform.score(train.X, train.Y, "cqr", trained_model=NULL, quantile=1-alpha)
t.mdl = train.score$model

########################################
## calibration 
########################################
calib.data = data.gen.ate(n,p,Gamma,beta,alpha0,obs=TRUE,u_dim)
calib.X = (calib.data$X[calib.data$T==1,])[1:n,]
calib.Y = (calib.data$Y1[calib.data$T==1])[1:n]
calib.U = (calib.data$U[calib.data$T==1,])[1:n,]  # U 是矩阵
calib.ex = (calib.data$ex[calib.data$T==1])[1:n]
n_calib = length(calib.Y)

# lower and upper bounds of weight function（原有方法）
calib.lx = pp * (1 + (1-calib.ex) / (calib.ex*Gamma))
calib.ux = pp * (1 + Gamma * (1-calib.ex) / (calib.ex))
calib.nc.w = pp / (calib.data$ex[calib.data$T==1])[1:n]

# non-conformity score on calibration data（原有方法）
calib.score = conform.score(calib.X, calib.Y, "cqr", trained_model=t.mdl, quantile=1-alpha)$score
calib.all = data.frame("score"=calib.score, "lx"=calib.lx, "ux"=calib.ux, "wx"=calib.nc.w, "ex" = calib.ex)
calib.all = calib.all[order(calib.all$score),]
rownames(calib.all) = 1:dim(calib.all)[1]

########################################
## generate test fold
########################################
test.data = data.gen.ate(n_test,p,Gamma,beta,alpha0,obs=FALSE,u_dim)
test.X = test.data$X
test.Y1 = test.data$Y1
test.U = test.data$U  # U 是矩阵
test.ex = test.data$ex
test.lx = pp*(1+ 1/Gamma * (1-test.ex)/test.ex)
test.ux = pp*(1+ Gamma* (1-test.ex)/(test.ex))

# predictions（原有方法）
test.pred = predict(t.mdl, test.X, quantile=c(alpha/2, 1-alpha/2)) 
# ✅ 兼容 grf 新旧版本
if (is.list(test.pred) && "predictions" %in% names(test.pred)) {
  test.pred <- test.pred$predictions
}

########################################
## 方法1: confounding-aware algorithm（原有）
########################################

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

########################################
## 方法2: confounding-unaware algorithm（原有）
########################################

cat(" - Computing the vanilla weighted conformal inference...")

nc.sum = rep(0,n_calib)
nc.sum[1] = calib.all$wx[1]
for (k in 2:n_calib){
  nc.sum[k] = nc.sum[k-1] + calib.all$wx[k]
}

nc.test.lo = rep(0,n_test)
nc.test.hi = rep(0,n_test)
nc.test.weight = pp / test.ex

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
## 【新增】方法3: Proximal CSPCI method
########################################

cat(" - Computing the proximal conformal inference...")

# 【真实边界版本】：计算真实的 e(X,W) = P(T=1|X,W)
# 基于数学推导的 closed form 公式（完全向量化版本，无 for-loop）
compute_lx_ux_from_e_true <- function(X_mat, W_grid, pbar, Gamma, beta, alpha0, noise_level, eps = 1e-3) {
  n <- nrow(X_mat)
  K <- nrow(W_grid)
  if (n < 1 || K < 1) {
    stop("Empty X_mat or W_grid in compute_lx_ux_from_e_true")
  }
  
  # 提取 W_grid 的第一列（对应 U1）
  W1_grid <- W_grid[, 1]  # K × 1 向量
  
  # Step 1: 计算所有 x 的基础量（向量化，n × 1）
  logit_ex <- alpha0 + X_mat %*% beta
  ex <- exp(logit_ex) / (1 + exp(logit_ex))
  
  ax <- ex / (ex + Gamma * (1 - ex))
  bx <- ex / (ex + (1 - ex) / Gamma)
  
  s_x <- abs(1 + 0.5 * sin(2.5 * X_mat[, 1]))
  
  prop_x <- ex
  p_x <- (1 / (prop_x + (1 - prop_x) / Gamma) - 1) / 
         (1 / (prop_x + (1 - prop_x) / Gamma) - 1 / (prop_x + Gamma * (1 - prop_x)))
  t_x <- qnorm(1 - p_x / 2) * s_x
  
  # Step 2: 计算后验分布参数（向量化，n × 1）
  sigma_w_sq <- noise_level^2
  tau_sq <- 1 / (1 / sigma_w_sq + 1 / (s_x^2))
  tau <- sqrt(tau_sq)
  
  # Step 3: 使用矩阵广播计算所有 (x, w) 组合的 μ(x,w)
  # tau_sq 是 n × 1，W1_grid 是 K × 1
  # 使用 outer 得到 n × K 矩阵：mu_mat[i,k] = tau_sq[i] * W1_grid[k] / sigma_w_sq
  mu_mat <- outer(tau_sq / sigma_w_sq, W1_grid, "*")  # n × K
  
  # Step 4: 计算 z_upper 和 z_lower（矩阵广播）
  # t_x 是 n × 1，tau 是 n × 1，mu_mat 是 n × K
  # 需要将 t_x 和 tau 扩展到 n × K
  t_x_mat <- matrix(t_x, nrow = n, ncol = K)  # n × K（每列相同）
  tau_mat <- matrix(tau, nrow = n, ncol = K)   # n × K（每列相同）
  
  z_upper_mat <- (t_x_mat - mu_mat) / tau_mat  # n × K
  z_lower_mat <- (-t_x_mat - mu_mat) / tau_mat  # n × K
  
  # Step 5: 计算 q(x,w) 矩阵（n × K）
  q_mat <- 1 - pnorm(z_upper_mat) + pnorm(z_lower_mat)
  
  # Step 6: 计算 e(x,w) 矩阵（n × K）
  # ax 和 bx 是 n × 1，需要扩展到 n × K
  ax_mat <- matrix(ax, nrow = n, ncol = K)  # n × K（每列相同）
  bx_mat <- matrix(bx, nrow = n, ncol = K)  # n × K（每列相同）
  ex_mat <- ax_mat * q_mat + bx_mat * (1 - q_mat)
  
  # Step 7: 裁剪并取每行的 max/min
  ex_mat <- pmax(pmin(ex_mat, 1 - eps), eps)
  emax <- apply(ex_mat, 1, max)  # n × 1
  emin <- apply(ex_mat, 1, min)  # n × 1
  
  lx <- pbar / emax
  ux <- pbar / emin
  
  return(list(lx = lx, ux = ux))
}

# sample candidate W grid from training W (for robustness and efficiency)
# 为全部训练数据生成 proxy W（用于采样 W grid）
if (w_dim < u_dim) {
  train.W.full = train.data$U[, 1:w_dim] + matrix(rnorm(nrow(train.data$X) * w_dim) * noise_level, nrow=nrow(train.data$X), ncol=w_dim)
} else if (w_dim == u_dim) {
  train.W.full = train.data$U + matrix(rnorm(nrow(train.data$X) * u_dim) * noise_level, nrow=nrow(train.data$X), ncol=u_dim)
} else {
  extra_cols = w_dim - u_dim
  W_extra.full = train.data$U %*% M_extra + matrix(rnorm(nrow(train.data$X) * extra_cols) * noise_level, nrow=nrow(train.data$X), ncol=extra_cols)
  train.W.full = cbind(
    train.data$U + matrix(rnorm(nrow(train.data$X) * u_dim) * noise_level, nrow=nrow(train.data$X), ncol=u_dim),
    W_extra.full
  )
}

set.seed(seed + 123)
w_grid_size <- min(50, nrow(train.W.full))
W.grid <- train.W.full[sample(nrow(train.W.full), w_grid_size), , drop = FALSE]

# Step 2: 为校准集计算真实的 proxy-based bounds
# 使用真实的 e(X,W) 公式，不依赖 Gamma（但需要 Gamma 来计算 e(X,W) 本身）
prox_bounds_calib <- compute_lx_ux_from_e_true(calib.X, W.grid, pbar = pp, Gamma = Gamma, 
                                                beta = beta, alpha0 = alpha0, noise_level = noise_level)
calib.lx.prox <- prox_bounds_calib$lx
calib.ux.prox <- prox_bounds_calib$ux

# Step 3: 对测试集计算真实的 proximal bounds
prox_bounds_test <- compute_lx_ux_from_e_true(test.X, W.grid, pbar = pp, Gamma = Gamma,
                                               beta = beta, alpha0 = alpha0, noise_level = noise_level)
test.lx.prox <- prox_bounds_test$lx
test.ux.prox <- prox_bounds_test$ux

# Step 4: 使用与 confounding-aware 相同的加权 conformal 流程（仅替换 l/u）
prox.sum.num = rep(0, n_calib)
prox.sum.den = rep(0, n_calib)
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

# 评估覆盖率
prox.cover = (prox.test.lo <= test.Y1) * (prox.test.hi >= test.Y1)

cat("Done.\n")

########################################
# output summary of test（输出三种方法的结果）
########################################
res = data.frame(
  # 原有的两种方法
  "c.cov" = mean(c.cover), 
  "c.len" = mean(c.test.hi-c.test.lo),
  "nc.cov" = mean(nc.cover), 
  "nc.len" = mean(nc.test.hi-nc.test.lo),
  # 【新增】proximal 方法
  "prox.cov" = mean(prox.cover),
  "prox.len" = mean(prox.test.hi - prox.test.lo),
  # 其他信息
  "n" = n, 
  "p" = p, 
  "u_dim" = u_dim,
  "w_dim" = w_dim,
  "n_calib" = n_calib,
  "gamma" = Gamma, 
  "alpha" = alpha, 
  "seed" = seed, 
  "method" = "marginal_plus_proximal"
)

save.path = file.path(out_dir, paste0("pred_marginal_prox_p_",p,"_n_",n,"_wdim_",w_dim,"_alpha_",alpha_ind,"_gamma_",Gamma_ind,"_seed_",seed,".csv"))
write.csv(res, save.path)

cat(paste(" - Results saved. Coverages: robust=", round(mean(c.cover),3), 
          ", unaware=", round(mean(nc.cover),3),
          ", proximal=", round(mean(prox.cover),3), "\n", sep=""))

