#!/usr/bin/env Rscript
########################################
## input configurations；参数设置
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
cat(paste(" - Running the script with marginally-valid algorithm and ground truth, alpha ", alpha, ", Gamma ",Gamma,
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
## Parameter； 数据生成参数
########################################
alpha0 = 0
n_test = 500  # 增加测试样本量以提高覆盖率估计的稳定性
u_dim = 20  # U 的类别数（离散版本）
x_levels = 10  # X 的类别数（仅用于离散版本）
y_levels = 200  # Y1 的类别数（仅用于离散版本）
beta = matrix(c(-0.531,0.126,-0.312,0.018,rep(0,p-4)), nrow=p)
# generate true probability of treatment
pp = mean(data.gen.ate(n*1000, p, Gamma, beta, alpha0, obs=FALSE, u_dim, u_dim, x_levels, y_levels)$T)


set.seed(seed)
########################################
## fit on the training fold

# 仅使用 T=1 的样本训练 CQR 模型
# 训练 nonconformity score 函数
########################################
train.data = data.gen.ate(n, p, Gamma, beta, alpha0, obs=TRUE, u_dim, u_dim, x_levels, y_levels)
train.X = (train.data$X[train.data$T==1,])[1:n,]
train.Y = (train.data$Y1[train.data$T==1])[1:n]
# 对离散Y添加小的随机噪声，使其更"连续"，便于quantile_forest预测分位数
# 噪声很小（±0.1），不会显著改变分布，但能让RF更好地学习分位数
set.seed(seed + 1000)  # 使用不同的seed避免影响数据生成
train.Y_continuous = train.Y + runif(length(train.Y), -0.1, 0.1)
# train the nonconformity score function
train.score = conform.score(train.X, train.Y_continuous, "cqr", trained_model=NULL, quantile=1-alpha)
t.mdl = train.score$model


########################################
## calibration；也是只有T=1
########################################
calib.data = data.gen.ate(n, p, Gamma, beta, alpha0, obs=TRUE, u_dim, u_dim, x_levels, y_levels)
calib.X = (calib.data$X[calib.data$T==1,])[1:n,]
calib.Y = (calib.data$Y1[calib.data$T==1])[1:n]
# 对校准数据的Y也添加相同的噪声
set.seed(seed + 2000)
calib.Y_continuous = calib.Y + runif(length(calib.Y), -0.1, 0.1)
calib.ex = (calib.data$ex[calib.data$T==1])[1:n]
n_calib = length(calib.Y)

# lower and upper bounds of weight function
calib.lx = pp * (1 + (1-calib.ex) / (calib.ex*Gamma)) 
calib.ux = pp * (1 + Gamma * (1-calib.ex) / (calib.ex))
calib.nc.w = pp / (calib.data$ex[calib.data$T==1])[1:n] # confounding-unaware weight

# 安全截断: 防止极端权重摧毁结果（特别是 N=2000 时）
# 使用校准集的 99% 分位数作为天花板
clip_val_lx <- quantile(calib.lx, 0.99)
clip_val_ux <- quantile(calib.ux, 0.99)
calib.lx <- pmin(calib.lx, clip_val_lx)
calib.ux <- pmin(calib.ux, clip_val_ux)

# non-conformity score on calibration data
# 使用训练数据训练好的CQR模型计算nonconformity score；并且按照score排序
# 用Y_continuous预测分位数，也用连续Y计算score（避免离散ties问题）
calib.score = conform.score(calib.X, calib.Y_continuous, "cqr", trained_model=t.mdl, quantile=1-alpha, Y_original=calib.Y_continuous)$score
calib.all = data.frame("score"=calib.score, "lx"=calib.lx, "ux"=calib.ux, "wx"=calib.nc.w, "ex" = calib.ex)
calib.all = calib.all[order(calib.all$score),]
rownames(calib.all) = 1:dim(calib.all)[1]

########################################
## generate test fold；包含所有的T
########################################
test.data = data.gen.ate(n_test, p, Gamma, beta, alpha0, obs=FALSE, u_dim, u_dim, x_levels, y_levels)
test.X = test.data$X
test.Y1 = test.data$Y1
test.ex = test.data$ex
test.lx = pp*(1+ 1/Gamma * (1-test.ex)/test.ex) # 真实边界
test.ux = pp*(1+ Gamma* (1-test.ex)/(test.ex))

# 截断测试集权重（使用与校准集相同的截断值）
test.lx <- pmin(test.lx, clip_val_lx)
test.ux <- pmin(test.ux, clip_val_ux)

test.pred = predict(t.mdl, test.X, quantile=c(alpha/2, 1-alpha/2)) 
# 兼容 grf 新旧版本
if (is.list(test.pred) && "predictions" %in% names(test.pred)) {
  test.pred <- test.pred$predictions
}

########################################
## the confounding-aware algorithm  
########################################

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

########################################
## the confounding-unaware algorithm  
########################################

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

########################################
# output summary of test   
########################################
res = data.frame("c.cov" = mean(c.cover), "c.len" = mean(c.test.hi-c.test.lo),
                 "nc.cov" = mean(nc.cover), "nc.len" = mean(nc.test.hi-nc.test.lo), 
                 "n" = n, "p" = p, "n_calib" = n_calib, "n_test" = n_test,
                 "gamma" = Gamma, "alpha" = alpha, "seed" = seed, 
                 "data_type" = data_type, "method" = "marginal",
                 "rf_min_node_size" = 5, "y_levels" = y_levels,
                 "note" = "n_test=500, min.node.size=5")

save.path = paste(out_dir, "pred_marginal_p_",p,"_n_",n,"_alpha_",alpha_ind,"_gamma_",Gamma_ind,"_seed_",seed,"_",data_type,".csv",sep='')
write.csv(res, file = save.path)