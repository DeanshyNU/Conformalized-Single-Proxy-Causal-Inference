
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
  source("../utils/util_ate_discrete.R")
  data_type <- "discrete"
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
n_test = 2000
u_dim = 20  # U 的类别数（离散版本）或维度数（连续版本）
w_dim = u_dim  # W 的维度（默认等于 u_dim，用于数据生成一致性）
beta = matrix(c(-0.531,0.126,-0.312,0.018,rep(0,p-4)), nrow=p)
noise_level = 0.5
x_levels = 10  # X 的类别数（仅用于离散版本，增加到10以更好地体现方法区别）
y_levels = 200  # Y1 的类别数（仅用于离散版本）
# generate true probability of treatment
  pp = mean(data.gen.ate(n*1000, p, Gamma, beta, alpha0, obs=FALSE, u_dim, w_dim, x_levels, y_levels)$T)


set.seed(seed)
########################################
## fit on the training fold
########################################
  train.data = data.gen.ate(n, p, Gamma, beta, alpha0, obs=TRUE, u_dim, w_dim, x_levels, y_levels)

train.X = (train.data$X[train.data$T==1,])[1:n,]
train.Y = (train.data$Y1[train.data$T==1])[1:n]
# 对离散Y添加小的随机噪声，使 score 分布更平滑（减少 ties）
set.seed(seed + 1000)
train.Y_continuous = train.Y + runif(length(train.Y), -0.1, 0.1)
# train the nonconformity score function
train.score = conform.score(train.X, train.Y_continuous, "cqr", trained_model=NULL, quantile=1-alpha)
t.mdl = train.score$model
# estimate hat{e}(x)
hat.p = mean(train.data$T)
e.model = regression_forest(train.data$X, train.data$T, num.threads = 1)

########################################
## calibration 
########################################
  calib.data = data.gen.ate(n, p, Gamma, beta, alpha0, obs=TRUE, u_dim, w_dim, x_levels, y_levels)
calib.X = (calib.data$X[calib.data$T==1,])[1:n,]
calib.Y = (calib.data$Y1[calib.data$T==1])[1:n]
set.seed(seed + 2000)
calib.Y_continuous = calib.Y + runif(length(calib.Y), -0.1, 0.1)
calib.ex = predict(e.model, newdata=calib.X)$predictions  
n_calib = length(calib.Y)

# 数值保护：避免 e(x) 过于极端导致权重爆炸
eps_ex = 1e-3
calib.ex = pmax(pmin(calib.ex, 1 - eps_ex), eps_ex)

# lower and upper bounds of weight function
calib.lx_raw = hat.p * (1+ (1-calib.ex) / (calib.ex*Gamma))
calib.ux_raw = hat.p * (1+ Gamma * (1-calib.ex) / (calib.ex))
calib.nc.w = hat.p / calib.ex

# 安全截断: 防止极端权重摧毁结果（特别是 N=2000 时）
# 使用校准集的 99% 分位数作为天花板
clip_val_lx <- quantile(calib.lx_raw, 0.99)
clip_val_ux <- quantile(calib.ux_raw, 0.99)
calib.lx <- pmin(calib.lx_raw, clip_val_lx)
calib.ux <- pmin(calib.ux_raw, clip_val_ux)

# non-conformity score
calib.score = conform.score(calib.X, calib.Y_continuous, "cqr", trained_model=t.mdl, quantile=1-alpha, Y_original=calib.Y_continuous)$score
calib.all = data.frame("score"=calib.score, "lx"=calib.lx, "ux"=calib.ux, "wx"=calib.nc.w, "ex" = calib.ex)
calib.all = calib.all[order(calib.all$score),]
rownames(calib.all) = 1:dim(calib.all)[1]

# calculate estimation error and actual gap
# 注意：离散版本也返回exu字段
calib.true.exu = calib.data$exu
calib.true.ex = calib.data$ex
calib.true.wx = pp / calib.true.exu
gap = (mean(pmax(calib.lx_raw - calib.true.wx, 0)) + mean(pmax(calib.true.wx - calib.ux_raw, 0)) + 1/nrow(calib.X) * mean( calib.true.wx * max(pmax(calib.true.wx - calib.ux_raw, 0))) ) * max(1/calib.lx_raw)
l_err = mean(abs(calib.lx_raw - pp * (1 + (1-calib.true.ex)/(calib.true.ex*Gamma))))
u_err = mean(abs(calib.ux_raw - pp * (1 + Gamma * (1-calib.true.ex)/calib.true.ex)))
l_inv = mean(1/calib.lx_raw)

########################################
## generate test fold
########################################
  test.data = data.gen.ate(n_test, p, Gamma, beta, alpha0, obs=FALSE, u_dim, w_dim, x_levels, y_levels)
test.X = test.data$X
test.Y1 = test.data$Y1
test.ex = predict(e.model, newdata=test.X)$predictions 
test.ex = pmax(pmin(test.ex, 1 - eps_ex), eps_ex)
test.lx_raw = hat.p*(1+ (1-test.ex)/(test.ex*Gamma))
test.ux_raw = hat.p*(1+ Gamma* (1-test.ex)/(test.ex))
test.lx = pmin(test.lx_raw, clip_val_lx)
test.ux = pmin(test.ux_raw, clip_val_ux)
test.wx = hat.p / test.ex
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
c.size = mean(pmax(0, floor(c.test.hi) - ceiling(c.test.lo) + 1))

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
nc.size = mean(pmax(0, floor(nc.test.hi) - ceiling(nc.test.lo) + 1))

cat("Done.\n")

########################################
# output summary of test   
########################################
res = data.frame("c.cov" = mean(c.cover), "c.len" = mean(c.test.hi-c.test.lo),
                 "c.size" = c.size,
                 "nc.cov" = mean(nc.cover), "nc.len" = mean(nc.test.hi-nc.test.lo),
                 "nc.size" = nc.size,
                 "gap" = gap, "l_err" = l_err, "u_err" = u_err, "l_inv" = l_inv,
                 "n" = n, "p" = p, "n_calib" = n_calib, "n_test" = n_test,
                 "gamma" = Gamma, "alpha" = alpha, "seed" = seed, 
                 "data_type" = data_type, "method" = "marginal_est",
                 "rf_min_node_size" = 5, "y_levels" = y_levels,
                 "note" = "n_test=2000, min.node.size=5, jitter=U(-0.1,0.1), ex_clip=1e-3, weight_clip=99%")

save.path = paste(out_dir, "pred_marginal_est_p_",p,"_n_",n,"_alpha_",alpha_ind,"_gamma_",Gamma_ind,"_seed_",seed,"_",data_type,".csv",sep='')
write.csv(res, file = save.path)