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
discrete_ind <- ifelse(length(args) >= 7, as.integer(args[7]), 1)  # 0=continuous, 1=discrete (default: discrete)

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
if (discrete_ind == 1) {
  source("../utils/util_ate_discrete.R")
  data_type <- "discrete"
} else {
  source("../utils/util_ate_multi.R")
  data_type <- "continuous"
}
cat(paste(" - Running pred_mgn_prox_est: estimated bounds (robust + unaware + proximal), alpha", alpha, 
          ", Gamma", Gamma, ", n", n, ", p", p, ", w_dim", w_dim, ", seed", seed, ", type", data_type, "\n"), sep = '')

########################################
## Output directory（估计边界版，按 w_dim 区分）
########################################
base_out_dir <- "/projects/p32685/cfsensitivity_results/simulation_prox_est/"
out_dir <- file.path(base_out_dir, paste0("wdim_", w_dim))
if(!dir.exists(out_dir)){
  dir.create(out_dir, recursive = TRUE)
}

########################################
## Parameter
########################################
alpha0 = 0
n_test = 500
u_dim = 20  # U 的类别数（离散版本）或维度数（连续版本）
beta = matrix(c(-0.531,0.126,-0.312,0.018,rep(0,p-4)), nrow=p)
noise_level = 0.5
x_levels = 5  # X 的类别数（仅用于离散版本）
y_levels = 100  # Y1 的类别数（仅用于离散版本）
# generate true probability of treatment
if (discrete_ind == 1) {
  pp = mean(data.gen.ate(n*1000, p, Gamma, beta, alpha0, obs=FALSE, u_dim, u_dim, x_levels, y_levels)$T)
} else {
  pp = mean(data.gen.ate(n*1000, p, Gamma, beta, alpha0, obs=FALSE)$T)
}

set.seed(seed)
########################################
## 生成所有数据（在训练模型之前，确保数据生成的一致性）
########################################
# 生成训练数据
if (discrete_ind == 1) {
  # Discrete version: use discrete data generation
  # 注意：对于 robust 和 unaware 方法，使用 u_dim, u_dim 确保与 pred_mgn_est.R 一致
  train.data = data.gen.ate(n, p, Gamma, beta, alpha0, obs=TRUE, u_dim, u_dim, x_levels, y_levels)
} else {
  # Continuous version: use continuous data generation
  train.data = data.gen.ate(n, p, Gamma, beta, alpha0, obs=TRUE, u_dim, u_dim, noise_level)
}

# 生成校准数据
if (discrete_ind == 1) {
  # 注意：对于 robust 和 unaware 方法，使用 u_dim, u_dim 确保与 pred_mgn_est.R 一致
  calib.data = data.gen.ate(n, p, Gamma, beta, alpha0, obs=TRUE, u_dim, u_dim, x_levels, y_levels)
} else {
  calib.data = data.gen.ate(n, p, Gamma, beta, alpha0, obs=TRUE, u_dim, u_dim, noise_level)
}

# 生成测试数据
if (discrete_ind == 1) {
  # 注意：对于 robust 和 unaware 方法，使用 u_dim, u_dim 确保与 pred_mgn_est.R 一致
  test.data = data.gen.ate(n_test, p, Gamma, beta, alpha0, obs=FALSE, u_dim, u_dim, x_levels, y_levels)
} else {
  test.data = data.gen.ate(n_test, p, Gamma, beta, alpha0, obs=FALSE, u_dim, u_dim, noise_level)
}

########################################
## 训练所有模型（在数据生成之后）
########################################
# 仅使用 treated 样本训练 CQR，与原方法一致
train.X = (train.data$X[train.data$T==1,])[1:n,]
train.Y = (train.data$Y1[train.data$T==1])[1:n]
# train the nonconformity score function（原有方法）
train.score = conform.score(train.X, train.Y, "cqr", trained_model=NULL, quantile=1-alpha)
t.mdl = train.score$model

# 估计 hat{p} 与 e(x)（估计边界版本）
hat.p = mean(train.data$T)
# 调试输出：检查训练数据
cat(" [debug] Training data check:\n")
cat("   hat.p =", hat.p, "\n")
cat("   train.data$T (first 10):", head(train.data$T, 10), "\n")
cat("   train.data$X (first row, first 5):", head(train.data$X[1,], 5), "\n")
e.model = regression_forest(train.data$X, train.data$T, num.threads = 1)
# 调试输出：检查 e.model 训练后的预测
train.ex.check = predict(e.model, newdata=head(train.data$X, 5))$predictions
cat("   e.model predictions (first 5):", train.ex.check, "\n")

# 【Proximal新增】：训练 proxy-based propensity（同样基于观测数据）
# 根据数据类型选择 W 的生成方式
if (discrete_ind == 1) {
  # Discrete version: W is already generated in data.gen.ate
  train.W.full = train.data$W
} else {
  # Continuous version: generate W from U + noise
  # 当 w_dim > u_dim 时，创建固定的混合矩阵（确保可重复性）
  if (w_dim > u_dim) {
    set.seed(seed + 999)  # 固定种子确保混合矩阵可重复
    extra_cols = w_dim - u_dim
    M_extra = matrix(rnorm(u_dim * extra_cols), nrow=u_dim, ncol=extra_cols)
  } else {
    M_extra = NULL
  }
  
  # 为全部训练数据生成 proxy W
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
}
train.XW.full = cbind(train.data$X, train.W.full)
e.model.prox = regression_forest(train.XW.full, train.data$T, num.threads = 1)

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
  # 使用真实的 min/max 而不是分位数（理论精确）
  emax <- apply(ex_mat, 1, max)
  emin <- apply(ex_mat, 1, min)
  lx <- pbar / emax
  ux <- pbar / emin
  # 调试输出：检查前几个样本的 e(x|X,W) 变化范围
  cat(" [debug] compute_lx_ux_from_e for first 3 samples:\n")
  for (i in 1:min(3, n)) {
    cat("   Sample", i, ": e(x|X,W) range = [", min(ex_mat[i,]), ",", max(ex_mat[i,]), "], diff =", max(ex_mat[i,]) - min(ex_mat[i,]), "\n")
    cat("     emax =", emax[i], ", emin =", emin[i], ", diff =", emax[i] - emin[i], "\n")
    cat("     lx =", lx[i], ", ux =", ux[i], ", ratio =", ux[i]/lx[i], "\n")
  }
  # 调试输出：检查整体统计
  cat(" [debug] e(x|X,W) statistics:\n")
  cat("   emax (min, max, mean):", min(emax), max(emax), mean(emax), "\n")
  cat("   emin (min, max, mean):", min(emin), max(emin), mean(emin), "\n")
  cat("   emax - emin (min, max, mean):", min(emax - emin), max(emax - emin), mean(emax - emin), "\n")
  cat("   ux/lx ratio (min, max, mean):", min(ux/lx), max(ux/lx), mean(ux/lx), "\n")
  return(list(lx = lx, ux = ux))
}

## sample candidate W grid from training W (for robustness and efficiency)
set.seed(seed + 123)
if (discrete_ind == 1) {
  # Discrete version: W is a vector, need to convert to matrix for grid sampling
  w_grid_size <- min(30, length(train.W.full))
  W.grid <- matrix(train.W.full[sample(length(train.W.full), w_grid_size)], ncol=1)
} else {
  # Continuous version: W is a matrix
  w_grid_size <- min(30, nrow(train.W.full))
  W.grid <- train.W.full[sample(nrow(train.W.full), w_grid_size), , drop = FALSE]
}

########################################
## calibration 
########################################
# 注意：calib.data 已经在上面生成，这里只需要提取数据
calib.X = (calib.data$X[calib.data$T==1,])[1:n,]
calib.Y = (calib.data$Y1[calib.data$T==1])[1:n]
calib.ex = predict(e.model, newdata=calib.X)$predictions
n_calib = length(calib.Y)
# 调试输出：检查校准数据的 e(x) 预测
cat(" [debug] Calibration e(x) check:\n")
cat("   calib.ex (first 5):", head(calib.ex, 5), "\n")
cat("   calib.ex (min, max, mean):", min(calib.ex), max(calib.ex), mean(calib.ex), "\n")

# lower and upper bounds of weight function（估计版使用 hat.p）
calib.lx = hat.p * (1 + (1-calib.ex) / (calib.ex*Gamma))
calib.ux = hat.p * (1 + Gamma * (1-calib.ex) / (calib.ex))
calib.nc.w = hat.p / calib.ex

# 调试输出：检查 Gamma 与三种权重是否区分开
cat(" [debug] Gamma =", Gamma, "\n")
cat(" [debug] calib.lx (head):", head(calib.lx, 5), "\n")
cat(" [debug] calib.ux (head):", head(calib.ux, 5), "\n")
cat(" [debug] calib.wx (head):", head(calib.nc.w, 5), "\n")

# non-conformity score on calibration data（原有方法）
calib.score = conform.score(calib.X, calib.Y, "cqr", trained_model=t.mdl, quantile=1-alpha)$score
calib.all = data.frame("score"=calib.score, "lx"=calib.lx, "ux"=calib.ux, "wx"=calib.nc.w, "ex" = calib.ex)
calib.all = calib.all[order(calib.all$score),]
rownames(calib.all) = 1:dim(calib.all)[1]

########################################
## generate test fold
########################################
# 注意：test.data 已经在上面生成，这里只需要提取数据
test.X = test.data$X
test.Y1 = test.data$Y1
test.ex = predict(e.model, newdata=test.X)$predictions
test.lx = hat.p*(1+ 1/Gamma * (1-test.ex)/test.ex)
test.ux = hat.p*(1+ Gamma* (1-test.ex)/(test.ex))

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
## 【新增】方法3: Proximal CSPCI method
########################################

cat(" - Computing the proximal conformal inference...")

# Step 1: 估计 proxy-based propensity score（用于构造 bounds）
# 训练阶段已经得到 e.model.prox
# Step 2: 为校准集与测试集计算 proximal bounds（不依赖 Gamma）
prox_bounds_calib <- compute_lx_ux_from_e(e.model.prox, calib.X, W.grid, pbar = hat.p)
calib.lx.prox <- prox_bounds_calib$lx
calib.ux.prox <- prox_bounds_calib$ux

# 【修复】按照 score 排序，使其与 calib.all 的顺序一致
# calib.score 已经在第167行计算，calib.all 在第169行按 score 排序
# 所以这里需要按照相同的顺序排序 proximal bounds
score_order <- order(calib.score)
calib.lx.prox <- calib.lx.prox[score_order]
calib.ux.prox <- calib.ux.prox[score_order]

# 调试输出：比较 proximal 和 unaware 的权重
cat(" [debug] Proximal vs Unaware weights comparison:\n")
cat("   calib.lx.prox (first 5):", head(calib.lx.prox, 5), "\n")
cat("   calib.ux.prox (first 5):", head(calib.ux.prox, 5), "\n")
cat("   calib.nc.w (first 5):", head(calib.nc.w, 5), "\n")
cat("   lx.prox vs wx ratio (first 5):", head(calib.lx.prox/calib.nc.w, 5), "\n")
cat("   ux.prox vs wx ratio (first 5):", head(calib.ux.prox/calib.nc.w, 5), "\n")
cat("   lx.prox vs wx ratio (mean, sd):", mean(calib.lx.prox/calib.nc.w), sd(calib.lx.prox/calib.nc.w), "\n")
cat("   ux.prox vs wx ratio (mean, sd):", mean(calib.ux.prox/calib.nc.w), sd(calib.ux.prox/calib.nc.w), "\n")

prox_bounds_test <- compute_lx_ux_from_e(e.model.prox, test.X, W.grid, pbar = hat.p)
test.lx.prox <- prox_bounds_test$lx
test.ux.prox <- prox_bounds_test$ux

# Step 3: 使用与 confounding-aware 相同的加权 conformal 流程（仅替换 l/u）
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

# 评估覆盖率（与原流程一致）
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
  "data_type" = data_type,
  "method" = "marginal_plus_proximal_est"
)

save.path = file.path(out_dir, paste0("pred_marginal_prox_p_",p,"_n_",n,"_wdim_",w_dim,"_alpha_",alpha_ind,"_gamma_",Gamma_ind,"_seed_",seed,"_",data_type,".csv"))
write.csv(res, save.path)

cat(paste(" - Results saved. Coverages: robust=", round(mean(c.cover),3), 
          ", unaware=", round(mean(nc.cover),3),
          ", proximal=", round(mean(prox.cover),3), "\n", sep=""))

