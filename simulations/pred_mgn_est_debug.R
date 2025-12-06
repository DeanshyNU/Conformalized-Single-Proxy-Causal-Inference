#!/usr/bin/env Rscript
########################################
## input configurations
########################################
args <- commandArgs(trailingOnly = TRUE)
p <- as.integer(args[1])
n <- as.integer(args[2])
alpha_ind <- as.integer(args[3])
Gamma_ind <- as.integer(args[4])
seed <- as.integer(args[5])
discrete_ind <- ifelse(length(args) >= 6, as.integer(args[6]), 0)  # 0=continuous, 1=discrete (default: continuous)

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
  source("../utils/util_ate.R")
  data_type <- "continuous"
}
cat(paste(" - Running the script with marginally-valid algorithm and estimated bounds, alpha ", alpha, ", Gamma ",Gamma,
          ", n ", n, ", p ", p, ", seed ", seed, ", type ", data_type, "\n"), sep = '')

########################################
## Output direcroty
########################################
if(!dir.exists("../results")){
  dir.create("../results")
}
out_dir <- "../results/simulation/"
if(!dir.exists(out_dir)){
  dir.create(out_dir)
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
## fit on the training fold
########################################
if (discrete_ind == 1) {
  train.data = data.gen.ate(n, p, Gamma, beta, alpha0, obs=TRUE, u_dim, u_dim, x_levels, y_levels)
} else {
  train.data = data.gen.ate(n, p, Gamma, beta, alpha0, obs=TRUE)
}

# 调试输出：检查数据生成的overlap情况
if (discrete_ind == 1) {
  cat(" [debug] Data generation check:\n")
  cat("   True e(x) (prop.x) distribution:\n")
  cat("     min:", min(train.data$ex), ", max:", max(train.data$ex), "\n")
  cat("     mean:", mean(train.data$ex), ", sd:", sd(train.data$ex), "\n")
  cat("     < 0.1:", sum(train.data$ex < 0.1), ", > 0.9:", sum(train.data$ex > 0.9), "\n")
  cat("   Treatment distribution:\n")
  cat("     T=0:", sum(train.data$T == 0), ", T=1:", sum(train.data$T == 1), "\n")
  cat("     hat.p =", mean(train.data$T), "\n")
  # 检查是否有完全分离：某些X组合只有T=0或只有T=1
  # 对于离散数据，检查前几列X的组合
  if (p >= 1) {
    x1_unique = unique(train.data$X[, 1])
    perfect_sep_count = 0
    for (x1_val in x1_unique) {
      idx = train.data$X[, 1] == x1_val
      t_vals = train.data$T[idx]
      if (length(unique(t_vals)) == 1) {
        perfect_sep_count = perfect_sep_count + 1
      }
    }
    cat("   Perfect separation (X[,1] only):", perfect_sep_count, "out of", length(x1_unique), "unique X[,1] values\n")
  }
}

train.X = (train.data$X[train.data$T==1,])[1:n,]
train.Y = (train.data$Y1[train.data$T==1])[1:n]
# train the nonconformity score function
train.score = conform.score(train.X, train.Y, "cqr", trained_model=NULL, quantile=1-alpha)
t.mdl = train.score$model
# estimate hat{e}(x)
hat.p = mean(train.data$T)
# 方案三：限制模型复杂度，防止过拟合（离散数据需要更大的 min.node.size）
# 增大 min.node.size 可以强制模型更保守，避免学到极端的 e(x)
if (discrete_ind == 1) {
  e.model = regression_forest(train.data$X, train.data$T, num.threads = 1, min.node.size = 50)
} else {
  e.model = regression_forest(train.data$X, train.data$T, num.threads = 1)
}

########################################
## calibration 
########################################
if (discrete_ind == 1) {
  calib.data = data.gen.ate(n, p, Gamma, beta, alpha0, obs=TRUE, u_dim, u_dim, x_levels, y_levels)
} else {
  calib.data = data.gen.ate(n, p, Gamma, beta, alpha0, obs=TRUE)
}
calib.X = (calib.data$X[calib.data$T==1,])[1:n,]
calib.Y = (calib.data$Y1[calib.data$T==1])[1:n]
calib.ex = predict(e.model, newdata=calib.X)$predictions
# 方案一：强制截断 e(x)，防止极端值导致权重失效
calib.ex = pmax(0.05, pmin(0.95, calib.ex))
n_calib = length(calib.Y)

# 调试输出：检查 e(x) 的分布，看是否极端化
cat(" [debug] RF e(x) distribution:\n")
cat("   min:", min(calib.ex), ", max:", max(calib.ex), "\n")
cat("   mean:", mean(calib.ex), ", sd:", sd(calib.ex), "\n")
cat("   < 0.1:", sum(calib.ex < 0.1), ", > 0.9:", sum(calib.ex > 0.9), "\n")
cat("   < 0.01:", sum(calib.ex < 0.01), ", > 0.99:", sum(calib.ex > 0.99), "\n")

# lower and upper bounds of weight function
calib.lx = hat.p * (1+ (1-calib.ex) / (calib.ex*Gamma))
calib.ux = hat.p * (1+ Gamma * (1-calib.ex) / (calib.ex))
calib.nc.w = hat.p / calib.ex

# 调试输出：检查权重是否成比例
cat(" [debug] Gamma =", Gamma, "\n")
cat(" [debug] First 5 weights comparison:\n")
cat("   calib.lx (head):", head(calib.lx, 5), "\n")
cat("   calib.wx (head):", head(calib.nc.w, 5), "\n")
cat("   ratio lx/wx (head):", head(calib.lx/calib.nc.w, 5), "\n")
cat("   ratio lx/wx (mean):", mean(calib.lx/calib.nc.w), ", sd:", sd(calib.lx/calib.nc.w), "\n")

# non-conformity score
calib.score = conform.score(calib.X, calib.Y, "cqr", trained_model=t.mdl, quantile=1-alpha)$score
calib.all = data.frame("score"=calib.score, "lx"=calib.lx, "ux"=calib.ux, "wx"=calib.nc.w, "ex" = calib.ex)
calib.all = calib.all[order(calib.all$score),]
rownames(calib.all) = 1:dim(calib.all)[1]

# calculate estimation error and actual gap
# 注意：离散版本也返回exu字段
calib.true.exu = calib.data$exu
calib.true.ex = calib.data$ex
calib.true.wx = pp / calib.true.exu
gap = (mean(pmax(calib.lx - calib.true.wx, 0)) + mean(pmax(calib.true.wx - calib.ux, 0)) + 1/nrow(calib.X) * mean( calib.true.wx * max(pmax(calib.true.wx - calib.ux, 0))) ) * max(1/calib.lx)
l_err = mean(abs(calib.lx - pp * (1 + (1-calib.true.ex)/(calib.true.ex*Gamma))))
u_err = mean(abs(calib.ux - pp * (1 + Gamma * (1-calib.true.ex)/calib.true.ex)))
l_inv = mean(1/calib.lx)

########################################
## generate test fold
########################################
if (discrete_ind == 1) {
  test.data = data.gen.ate(n_test, p, Gamma, beta, alpha0, obs=FALSE, u_dim, u_dim, x_levels, y_levels)
} else {
  test.data = data.gen.ate(n_test, p, Gamma, beta, alpha0, obs=FALSE)
}
test.X = test.data$X
test.Y1 = test.data$Y1
test.ex = predict(e.model, newdata=test.X)$predictions
# 方案一：强制截断 e(x)，防止极端值导致权重失效
test.ex = pmax(0.05, pmin(0.95, test.ex))
test.lx = hat.p*(1+ (1-test.ex)/(test.ex*Gamma))
test.ux = hat.p*(1+ Gamma* (1-test.ex)/(test.ex))
test.wx = hat.p / test.ex
# 调试输出：检查测试集权重
cat(" [debug] Test weights comparison:\n")
cat("   test.lx (head):", head(test.lx, 5), "\n")
cat("   test.wx (head):", head(test.wx, 5), "\n")
cat("   ratio lx/wx (head):", head(test.lx/test.wx, 5), "\n")
cat("   ratio lx/wx (mean):", mean(test.lx/test.wx), ", sd:", sd(test.lx/test.wx), "\n")
test.pred = predict(t.mdl, test.X, quantile=c(alpha/2, 1-alpha/2)) 
# ✅ 兼容 grf 新旧版本
if (is.list(test.pred) && "predictions" %in% names(test.pred)) {
  test.pred <- test.pred$predictions
} 

########################################
## the confounding-aware algorithm  
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
## the confounding-unaware algorithm  
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
# output summary of test   
########################################
res = data.frame("c.cov" = mean(c.cover), "c.len" = mean(c.test.hi-c.test.lo),
                 "nc.cov" = mean(nc.cover), "nc.len" = mean(nc.test.hi-nc.test.lo), 
                 "gap" = gap, "l_err" = l_err, "u_err" = u_err, "l_inv" = l_inv,
                 "n" = n, "p" = p, "n_calib" = n_calib,
                 "gamma" = Gamma, "alpha" = alpha, "seed" = seed, 
                 "data_type" = data_type, "method" = "marginal_est")

save.path = paste(out_dir, "pred_marginal_est_p_",p,"_n_",n,"_alpha_",alpha_ind,"_gamma_",Gamma_ind,"_seed_",seed,"_",data_type,".csv",sep='')
write.csv(res, save.path)