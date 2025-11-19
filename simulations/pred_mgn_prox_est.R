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
cat(paste(" - Running pred_mgn_prox_est: estimated bounds (robust + unaware + proximal), alpha", alpha, 
          ", Gamma", Gamma, ", n", n, ", p", p, ", w_dim", w_dim, ", seed", seed, "\n"), sep = '')

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
u_dim = 20  # U 固定为 20 维
beta = matrix(c(-0.531,0.126,-0.312,0.018,rep(0,p-4)), nrow=p)
set.seed(seed)
########################################
## fit on the training fold
## 【新增】：生成带 proxy W 的数据（U 是 20 维矩阵）
########################################
train.data = data.gen.ate(n,p,Gamma,beta,alpha0,obs=TRUE,u_dim)

# 仅使用 treated 样本训练 CQR，与原方法一致
train.X = (train.data$X[train.data$T==1,])[1:n,]
train.Y = (train.data$Y1[train.data$T==1])[1:n]
train.U = (train.data$U[train.data$T==1,])[1:n,]  # U 现在是 n x 20 的矩阵

# 【Proximal新增】：根据 w_dim 生成不同维度的 proxy W
noise_level = 0.5
if (w_dim < u_dim) {
  # Case 1: W 维度小于 U（18维）- 只取 U 的前 w_dim 列
  train.W = train.U[, 1:w_dim] + matrix(rnorm(n * w_dim) * noise_level, nrow=n, ncol=w_dim)
} else if (w_dim == u_dim) {
  # Case 2: W 维度等于 U（20维）
  train.W = train.U + matrix(rnorm(n * u_dim) * noise_level, nrow=n, ncol=u_dim)
} else {
  # Case 3: W 维度大于 U（22维）- U + 额外噪声列
  extra_cols = w_dim - u_dim
  train.W = cbind(
    train.U + matrix(rnorm(n * u_dim) * noise_level, nrow=n, ncol=u_dim),
    matrix(rnorm(n * extra_cols), nrow=n, ncol=extra_cols)
  )
}

# train the nonconformity score function（原有方法）
train.score = conform.score(train.X, train.Y, "cqr", trained_model=NULL, quantile=1-alpha)
t.mdl = train.score$model

# 估计 hat{p} 与 e(x)（估计边界版本）
hat.p = mean(train.data$T)
e.model = regression_forest(train.data$X, train.data$T, num.threads = 1)

# 【Proximal新增】：训练 proxy-based outcome models（使用 (X,W)）
train.XW = cbind(train.X, train.W)
t.mdl.prox = quantile_forest(train.XW, train.Y, num.threads = 1)

# 【Proximal新增】：训练 proxy-based propensity（同样基于观测数据）
# 为全部训练数据生成 proxy W
if (w_dim < u_dim) {
  train.W.full = train.data$U[, 1:w_dim] + matrix(rnorm(nrow(train.data$X) * w_dim) * noise_level, nrow=nrow(train.data$X), ncol=w_dim)
} else if (w_dim == u_dim) {
  train.W.full = train.data$U + matrix(rnorm(nrow(train.data$X) * u_dim) * noise_level, nrow=nrow(train.data$X), ncol=u_dim)
} else {
  extra_cols = w_dim - u_dim
  train.W.full = cbind(
    train.data$U + matrix(rnorm(nrow(train.data$X) * u_dim) * noise_level, nrow=nrow(train.data$X), ncol=u_dim),
    matrix(rnorm(nrow(train.data$X) * extra_cols), nrow=nrow(train.data$X), ncol=extra_cols)
  )
}
train.XW.full = cbind(train.data$X, train.W.full)
e.model.prox = regression_forest(train.XW.full, train.data$T, num.threads = 1)

########################################
## calibration 
########################################
calib.data = data.gen.ate(n,p,Gamma,beta,alpha0,obs=TRUE,u_dim)
calib.X = (calib.data$X[calib.data$T==1,])[1:n,]
calib.Y = (calib.data$Y1[calib.data$T==1])[1:n]
calib.U = (calib.data$U[calib.data$T==1,])[1:n,]  # U 是矩阵
calib.ex = predict(e.model, newdata=calib.X)$predictions
n_calib = length(calib.Y)

# 【Proximal新增】：为校准集生成 proxy W（与训练集相同逻辑）
if (w_dim < u_dim) {
  calib.W = calib.U[, 1:w_dim] + matrix(rnorm(n * w_dim) * noise_level, nrow=n, ncol=w_dim)
} else if (w_dim == u_dim) {
  calib.W = calib.U + matrix(rnorm(n * u_dim) * noise_level, nrow=n, ncol=u_dim)
} else {
  extra_cols = w_dim - u_dim
  calib.W = cbind(
    calib.U + matrix(rnorm(n * u_dim) * noise_level, nrow=n, ncol=u_dim),
    matrix(rnorm(n * extra_cols), nrow=n, ncol=extra_cols)
  )
}

# lower and upper bounds of weight function（估计版使用 hat.p）
calib.lx = hat.p * (1 + (1-calib.ex) / (calib.ex*Gamma))
calib.ux = hat.p * (1 + Gamma * (1-calib.ex) / (calib.ex))
calib.nc.w = hat.p / calib.ex

# non-conformity score on calibration data（原有方法）
calib.score = conform.score(calib.X, calib.Y, "cqr", trained_model=t.mdl, quantile=1-alpha)$score
calib.all = data.frame("score"=calib.score, "lx"=calib.lx, "ux"=calib.ux, "wx"=calib.nc.w, "ex" = calib.ex)
calib.all = calib.all[order(calib.all$score),]
rownames(calib.all) = 1:dim(calib.all)[1]

# 【Proximal新增】：计算 proxy-based predictions for calibration
calib.XW = cbind(calib.X, W = calib.W)
calib.pred.prox = predict(t.mdl.prox, calib.XW, quantile=c(alpha/2, 1-alpha/2))
if (is.list(calib.pred.prox) && "predictions" %in% names(calib.pred.prox)) {
  calib.pred.prox <- calib.pred.prox$predictions
}

########################################
## generate test fold
########################################
test.data = data.gen.ate(n_test,p,Gamma,beta,alpha0,obs=FALSE,u_dim)
test.X = test.data$X
test.Y1 = test.data$Y1
test.U = test.data$U  # U 是矩阵
test.ex = predict(e.model, newdata=test.X, num.threads=1)$predictions
test.lx = hat.p*(1+ 1/Gamma * (1-test.ex)/test.ex)
test.ux = hat.p*(1+ Gamma* (1-test.ex)/(test.ex))

# 【Proximal新增】：为测试集生成 proxy W（与训练集相同逻辑）
if (w_dim < u_dim) {
  test.W = test.U[, 1:w_dim] + matrix(rnorm(n_test * w_dim) * noise_level, nrow=n_test, ncol=w_dim)
} else if (w_dim == u_dim) {
  test.W = test.U + matrix(rnorm(n_test * u_dim) * noise_level, nrow=n_test, ncol=u_dim)
} else {
  extra_cols = w_dim - u_dim
  test.W = cbind(
    test.U + matrix(rnorm(n_test * u_dim) * noise_level, nrow=n_test, ncol=u_dim),
    matrix(rnorm(n_test * extra_cols), nrow=n_test, ncol=extra_cols)
  )
}

# predictions（原有方法）
test.pred = predict(t.mdl, test.X, quantile=c(alpha/2, 1-alpha/2)) 
# ✅ 兼容 grf 新旧版本
if (is.list(test.pred) && "predictions" %in% names(test.pred)) {
  test.pred <- test.pred$predictions
}

# 【Proximal新增】：proxy-based predictions
test.XW = cbind(test.X, W = test.W)
test.pred.prox = predict(t.mdl.prox, test.XW, quantile=c(alpha/2, 1-alpha/2))
if (is.list(test.pred.prox) && "predictions" %in% names(test.pred.prox)) {
  test.pred.prox <- test.pred.prox$predictions
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

# Step 1: 估计 proxy-based propensity score（用于构造 feasible bounds）
# 训练阶段已经得到 e.model.prox
# Step 2: 为校准集计算 proxy-based feasible ITE bounds
calib.ex.prox = predict(e.model.prox, calib.XW)$predictions

# 计算 feasible bounds（类似 lx/ux，但基于 proxy，使用 hat.p）
calib.lx.prox = hat.p * (1 + (1-calib.ex.prox) / (calib.ex.prox*Gamma))
calib.ux.prox = hat.p * (1 + Gamma * (1-calib.ex.prox) / (calib.ex.prox))

# Step 3: 计算 ITE 点估计（用于 nonconformity score）
# 简化版：直接用 prediction interval 的中点作为点估计
calib.ite.prox = (calib.pred.prox[,2] + calib.pred.prox[,1]) / 2

# Step 4: 计算 feasible ITE interval 的上下界
# 使用简化的 feasible set：[tau_min, tau_max]
# 这里用 proxy-based bounds 估计
calib.tau.min = calib.pred.prox[,1] - mean(calib.pred.prox[,2] - calib.pred.prox[,1]) * 
                (1 - calib.lx.prox / calib.ux.prox)
calib.tau.max = calib.pred.prox[,2] + mean(calib.pred.prox[,2] - calib.pred.prox[,1]) * 
                (calib.ux.prox / calib.lx.prox - 1)

# Step 5: 计算 distance-based nonconformity score
# score = distance from ITE point estimate to feasible interval
calib.score.prox = pmax(0, 
                        calib.tau.min - calib.ite.prox,  # 低于下界
                        calib.ite.prox - calib.tau.max)  # 高于上界

# Step 6: 计算 conformal quantile
q_alpha.prox = quantile(calib.score.prox, probs = 1 - alpha, type = 1)

# Step 7: 对测试集应用 proximal method
test.ex.prox = predict(e.model.prox, test.XW)$predictions

# 计算测试集的 feasible bounds
test.lx.prox = hat.p * (1 + (1-test.ex.prox) / (test.ex.prox*Gamma))
test.ux.prox = hat.p * (1 + Gamma * (1-test.ex.prox) / (test.ex.prox))

# 计算 base feasible interval
test.tau.min = test.pred.prox[,1] - mean(test.pred.prox[,2] - test.pred.prox[,1]) * 
               (1 - test.lx.prox / test.ux.prox)
test.tau.max = test.pred.prox[,2] + mean(test.pred.prox[,2] - test.pred.prox[,1]) * 
               (test.ux.prox / test.lx.prox - 1)

# 用 conformal quantile 扩张区间
prox.test.lo = test.tau.min - q_alpha.prox
prox.test.hi = test.tau.max + q_alpha.prox

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
"method" = "marginal_plus_proximal_est"
)

save.path = file.path(out_dir, paste0("pred_marginal_prox_p_",p,"_n_",n,"_wdim_",w_dim,"_alpha_",alpha_ind,"_gamma_",Gamma_ind,"_seed_",seed,".csv"))
write.csv(res, save.path)

cat(paste(" - Results saved. Coverages: robust=", round(mean(c.cover),3), 
          ", unaware=", round(mean(nc.cover),3),
          ", proximal=", round(mean(prox.cover),3), "\n", sep=""))

