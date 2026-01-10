#!/usr/bin/env Rscript
########################################
## input configurations
########################################
args <- commandArgs(trailingOnly = TRUE)
p <- as.integer(args[1])
n <- as.integer(args[2])
w_dim <- as.integer(args[3])     # W 的维度（离散类别数）
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
## load util functions (离散版本)
########################################
source("../utils/util_ate_discrete.R")
data_type <- "discrete"
cat(paste(" - Running pred_mgn_prox_est_discrete: estimated bounds (robust + unaware + proximal), alpha", alpha, 
          ", Gamma", Gamma, ", n", n, ", p", p, ", w_dim", w_dim, ", seed", seed, ", type", data_type, "\n"), sep = '')

########################################
## Output directory
########################################
if(!dir.exists("../results")){
  dir.create("../results")
}
out_dir <- "../results/simulation/"
if(!dir.exists(out_dir)){
  dir.create(out_dir)
}

########################################
## Parameter (离散版本参数)
########################################
alpha0 = 0
n_test = 2000
u_dim = 20       # U 的类别数（必须为偶数）
x_levels = 10     # X 的类别数
y_levels = 200   # Y1 的类别数
beta = matrix(c(-0.531,0.126,-0.312,0.018,rep(0,p-4)), nrow=p)

# generate true probability of treatment (使用离散版数据生成)
pp = mean(data.gen.ate(n*1000, p, Gamma, beta, alpha0, obs=FALSE, u_dim, w_dim, x_levels, y_levels)$T)

set.seed(seed)
########################################
## 生成所有数据（离散版本）
########################################
# 生成训练数据
train.data = data.gen.ate(n, p, Gamma, beta, alpha0, obs=TRUE, u_dim, w_dim, x_levels, y_levels)

# 生成校准数据
calib.data = data.gen.ate(n, p, Gamma, beta, alpha0, obs=TRUE, u_dim, w_dim, x_levels, y_levels)

# 生成测试数据
test.data = data.gen.ate(n_test, p, Gamma, beta, alpha0, obs=FALSE, u_dim, w_dim, x_levels, y_levels)

########################################
## 训练所有模型（离散版本 + Jittering）
########################################
# 仅使用 treated 样本训练 CQR
train.X = (train.data$X[train.data$T==1,])[1:n,]
train.Y = (train.data$Y1[train.data$T==1])[1:n]

# 【关键】Jittering: 对离散 Y 添加对称噪声 U[-0.5, 0.5]，打破 Ties
set.seed(seed + 1000)
train.Y.cont = train.Y + runif(length(train.Y), -0.1, 0.1)

# 使用连续化的 Y 训练 CQR
train.score = conform.score(train.X, train.Y.cont, "cqr", trained_model=NULL, quantile=1-alpha)
t.mdl = train.score$model

# 估计 hat{p} 与 e(x)
hat.p = mean(train.data$T)
e.model = regression_forest(train.data$X, train.data$T, num.threads = 1)

# 【Proximal】：训练 proxy-based propensity（使用离散 W）
# 离散版本：W 已由 data.gen.ate 生成，直接使用
train.W = train.data$W
# 确保 W 是矩阵格式
if (is.vector(train.W)) {
  train.W.mat = matrix(train.W, ncol=1)
} else {
  train.W.mat = train.W
}
train.XW = cbind(train.data$X, train.W.mat)
e.model.prox = regression_forest(train.XW, train.data$T, num.threads = 1)

########################################
# helper for proximal bounds (简化版，无调试输出)
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
  emax <- apply(ex_mat, 1, max)
  emin <- apply(ex_mat, 1, min)
  lx <- pbar / emax
  ux <- pbar / emin
  return(list(lx = lx, ux = ux))
}

########################################
# 【离散版关键】W.grid 全量枚举
########################################
# 离散 W：直接枚举所有可能的类别值 1:w_dim
W.grid <- matrix(seq_len(w_dim), ncol = 1)

########################################
## calibration (离散版本 + Jittering)
########################################
calib.X = (calib.data$X[calib.data$T==1,])[1:n,]
calib.Y = (calib.data$Y1[calib.data$T==1])[1:n]

# 【关键】校准集也添加对称噪声
set.seed(seed + 2000)
calib.Y.cont = calib.Y + runif(length(calib.Y), -0.1, 0.1)

# e(x) 预测 + 数值保护
calib.ex = predict(e.model, newdata=calib.X)$predictions
eps_ex = 1e-3
calib.ex = pmax(pmin(calib.ex, 1 - eps_ex), eps_ex)
n_calib = length(calib.Y)

# lower and upper bounds of weight function（估计版使用 hat.p）
calib.lx_raw = hat.p * (1 + (1-calib.ex) / (calib.ex*Gamma))
calib.ux_raw = hat.p * (1 + Gamma * (1-calib.ex) / (calib.ex))
calib.nc.w = hat.p / calib.ex

# 权重截断（99% 分位数）
clip_val_lx <- quantile(calib.lx_raw, 0.99)
clip_val_ux <- quantile(calib.ux_raw, 0.99)
calib.lx <- pmin(calib.lx_raw, clip_val_lx)
calib.ux <- pmin(calib.ux_raw, clip_val_ux)

# non-conformity score (使用 Jittered Y)
calib.score = conform.score(calib.X, calib.Y.cont, "cqr", trained_model=t.mdl, quantile=1-alpha, Y_original=calib.Y.cont)$score
calib.all = data.frame("score"=calib.score, "lx"=calib.lx, "ux"=calib.ux, "wx"=calib.nc.w, "ex" = calib.ex)
calib.all = calib.all[order(calib.all$score),]
rownames(calib.all) = 1:dim(calib.all)[1]

########################################
## generate test fold
########################################
test.X = test.data$X
test.Y1 = test.data$Y1
test.ex = predict(e.model, newdata=test.X)$predictions
test.ex = pmax(pmin(test.ex, 1 - eps_ex), eps_ex)

test.lx_raw = hat.p*(1+ 1/Gamma * (1-test.ex)/test.ex)
test.ux_raw = hat.p*(1+ Gamma* (1-test.ex)/(test.ex))
# 使用校准集的截断值
test.lx = pmin(test.lx_raw, clip_val_lx)
test.ux = pmin(test.ux_raw, clip_val_ux)

# predictions
test.pred = predict(t.mdl, test.X, quantile=c(alpha/2, 1-alpha/2)) 
if (is.list(test.pred) && "predictions" %in% names(test.pred)) {
  test.pred <- test.pred$predictions
}

########################################
## 方法1: confounding-aware algorithm
########################################
cat(" - Computing the robust weighted conformal inference...")

sum.num = rep(0,n_calib)
sum.den = rep(0,n_calib)
sum.num[1] = calib.all$lx[1]
sum.den[1] = calib.all$lx[1] + sum(calib.all$ux[2:n_calib])
for (k in 2:n_calib){
  sum.num[k] = sum.num[k-1] + calib.all$lx[k]
  sum.den[k] = sum.den[k-1] - calib.all$ux[k] + calib.all$lx[k] 
}

c.test.lo = rep(0,n_test)
c.test.hi = rep(0,n_test)
for (ii in 1:n_test){
  ratios = sum.num / (sum.den + test.ux[ii])
  kstar = min(which(ratios>1-alpha))
  v.kstar = calib.all$score[kstar]
  c.test.lo[ii] = test.pred[ii,1]-v.kstar
  c.test.hi[ii] = test.pred[ii,2]+v.kstar
}
c.cover = (c.test.lo <= test.Y1) * (c.test.hi >= test.Y1)

cat("Done.\n")

########################################
## 方法2: confounding-unaware algorithm
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
## 方法3: Proximal CSPCI method
########################################
cat(" - Computing the proximal conformal inference...")

# 为校准集计算 proximal bounds
# 需要获取校准集对应的 W
calib.W = (calib.data$W[calib.data$T==1])[1:n]
if (is.vector(calib.W)) {
  calib.W.mat = matrix(calib.W, ncol=1)
} else {
  calib.W.mat = calib.W
}

prox_bounds_calib <- compute_lx_ux_from_e(e.model.prox, calib.X, W.grid, pbar = hat.p)
calib.lx.prox_raw <- prox_bounds_calib$lx
calib.ux.prox_raw <- prox_bounds_calib$ux

# 权重截断（99% 分位数）- 与 robust 方法保持一致（在排序之前截断）
clip_val_lx_prox <- quantile(calib.lx.prox_raw, 0.99)
clip_val_ux_prox <- quantile(calib.ux.prox_raw, 0.99)
calib.lx.prox <- pmin(calib.lx.prox_raw, clip_val_lx_prox)
calib.ux.prox <- pmin(calib.ux.prox_raw, clip_val_ux_prox)

# 按照 score 排序
score_order <- order(calib.score)
calib.lx.prox <- calib.lx.prox[score_order]
calib.ux.prox <- calib.ux.prox[score_order]

# 为测试集计算 proximal bounds
prox_bounds_test <- compute_lx_ux_from_e(e.model.prox, test.X, W.grid, pbar = hat.p)
test.lx.prox <- prox_bounds_test$lx
test.ux.prox <- prox_bounds_test$ux
# 测试集也使用相同的截断值
test.lx.prox <- pmin(test.lx.prox, clip_val_lx_prox)
test.ux.prox <- pmin(test.ux.prox, clip_val_ux_prox)

# 使用与 confounding-aware 相同的加权 conformal 流程
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

prox.cover = (prox.test.lo <= test.Y1) * (prox.test.hi >= test.Y1)

cat("Done.\n")

########################################
# 【离散版关键】计算 Set Size 指标
########################################
calc_size <- function(lo, hi, y_lev) {
  # 截断到合法范围
  lo <- pmax(1, pmin(y_lev, lo))
  hi <- pmax(1, pmin(y_lev, hi))
  lo[!is.finite(lo)] <- NA
  hi[!is.finite(hi)] <- NA
  mean(pmax(0, floor(hi) - ceiling(lo) + 1), na.rm = TRUE)
}

c.size = calc_size(c.test.lo, c.test.hi, y_levels)
nc.size = calc_size(nc.test.lo, nc.test.hi, y_levels)
prox.size = calc_size(prox.test.lo, prox.test.hi, y_levels)

########################################
# output summary（包含离散专用指标）
########################################
res = data.frame(
  # confounding-aware
  "c.cov" = mean(c.cover), 
  "c.len" = mean(c.test.hi-c.test.lo),
  "c.size" = c.size,
  # confounding-unaware
  "nc.cov" = mean(nc.cover), 
  "nc.len" = mean(nc.test.hi-nc.test.lo),
  "nc.size" = nc.size,
  # proximal
  "prox.cov" = mean(prox.cover),
  "prox.len" = mean(prox.test.hi - prox.test.lo),
  "prox.size" = prox.size,
  # 其他信息
  "n" = n, 
  "p" = p, 
  "u_dim" = u_dim,
  "w_dim" = w_dim,
  "x_levels" = x_levels,
  "y_levels" = y_levels,
  "n_calib" = n_calib,
  "n_test" = n_test,
  "gamma" = Gamma, 
  "alpha" = alpha, 
  "seed" = seed,
  "data_type" = data_type,
  "method" = "marginal_plus_proximal_est_discrete",
  "note" = "jitter=U(-0.1,0.1), W_grid=full_enum, ex_clip=1e-3, weight_clip=99%, beta_w_U_enhanced=[-1.5,1.5]"
)

save.path = paste(out_dir, "pred_marginal_prox_p_",p,"_n_",n,"_wdim_",w_dim,"_alpha_",alpha_ind,"_gamma_",Gamma_ind,"_seed_",seed,"_",data_type,".csv",sep='')
write.csv(res, file = save.path)

cat(paste(" - Results saved. Coverages: robust=", round(mean(c.cover),3), 
          ", unaware=", round(mean(nc.cover),3),
          ", proximal=", round(mean(prox.cover),3), "\n", sep=""))
