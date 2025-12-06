############################################################
#####   discrete data generating process (softmax)    ######
# n: output sample size 
# p: covariate dimension 
# Gamma: confounding level
# beta, alpha0: linear coefficients (for compatibility, not directly used in discrete case)
# u_dim: number of discrete categories for U (default 20, U takes values {1, 2, ..., u_dim})
# w_dim: number of discrete categories for W (default 20, W takes values {1, 2, ..., w_dim})
# x_levels: number of discrete categories for X (default 20, X takes values {1, 2, ..., x_levels})
# y_levels: number of discrete categories for Y1 (default 100, Y1 takes values {1, 2, ..., y_levels})
# obs = TRUE generate observations (training data) of size n
# obs = FALSE generate all data of size n
############################################################

data.gen.ate <- function(n, p, Gamma, beta, alpha0=0, obs=TRUE, u_dim=20, w_dim=20, x_levels=5, y_levels=100, coeff_seed=12345){
  # X: discrete (parameters from Unif[0.1, 1] for each dimension)
  # For each dimension j, sample x_levels parameters from Unif[0.1, 1]
  # Normalize to probabilities and sample discrete values
  X = matrix(0, nrow=n, ncol=p)
  for (j in 1:p) {
    # Sample parameters from Unif[0.1, 1] for this dimension
    params_j = runif(x_levels, 0.1, 1)
    # Normalize to probability distribution
    probs_j = params_j / sum(params_j)
    # Sample n discrete values from this distribution
    X[, j] = sample(1:x_levels, size=n, replace=TRUE, prob=probs_j)
  }
  
  # Save current random seed state (if exists)
  if (exists(".Random.seed", envir = .GlobalEnv)) {
    old_seed = get(".Random.seed", envir = .GlobalEnv)
    seed_exists = TRUE
  } else {
    seed_exists = FALSE
  }
  
  # Use fixed seed to generate coefficients (ensures reproducibility)
  set.seed(coeff_seed)
  
  # Initialize coefficient matrices (from Unif[-0.5, 0.5])
  # For P(U|X): beta_u is (u_dim x (p+1)) matrix
  # Features: [1 (intercept), X (p)] = p + 1
  # Note: β_{u} coefficients must be non-zero (sampling from Unif[-0.5, 0.5] ensures high probability of non-zero)
  beta_u = matrix(runif(u_dim*(p+1), -0.5, 0.5), nrow=u_dim, ncol=p+1)
  
  # For P(W|U,X): beta_w is (w_dim x (p+2)) matrix
  # Features: [1 (intercept), U (1), X (p)] = 1 + 1 + p = p + 2
  # Note: β_{w,U} must be non-zero to ensure W is relevant to U
  beta_w = matrix(runif(w_dim*(p+2), -0.5, 0.5), nrow=w_dim, ncol=p+2)
  
  # For P(Y1|U,X): beta_y is (y_levels x (p+2)) matrix
  # Features: [1 (intercept), U (1), X (p)] = 1 + 1 + p = p + 2
  # Note: Y1 does NOT depend on A (treatment), only on U and X (for counterfactual prediction)
  beta_y = matrix(runif(y_levels*(p+2), -0.5, 0.5), nrow=y_levels, ncol=p+2)
  
  # For P(A|X): beta_a is (2 x (p+1)) matrix
  # Features: [1 (intercept), X (p)] = p + 1
  beta_a = matrix(runif(2*(p+1), -0.5, 0.5), nrow=2, ncol=p+1)
  
  # Restore random seed (if it existed before)
  if (seed_exists) {
    assign(".Random.seed", old_seed, envir = .GlobalEnv)
  }
  
  # 归一化X和U，将类别索引映射到[0,1]，避免logits过大导致softmax饱和
  # 将 1~x_levels 映射到 0~1
  X_normalized = (X - 1) / (x_levels - 1)
  
  # Generate U using softmax P(U|X)
  # U is a vector of length n, each element takes values {1, 2, ..., u_dim}
  U = numeric(n)
  for (i in 1:n) {
    # 使用归一化后的X计算logit，避免数值量级过大
    x_vec = c(1, X_normalized[i,])  # intercept + normalized X
    logits_u = beta_u %*% x_vec
    probs_u = exp(logits_u) / sum(exp(logits_u))
    U[i] = sample(1:u_dim, size=1, prob=probs_u)
  }
  
  # 归一化U，将 1~u_dim 映射到 0~1
  U_normalized = (U - 1) / (u_dim - 1)
  
  # Generate W using softmax P(W|U,X)
  # W is a vector of length n, each element takes values {1, 2, ..., w_dim}
  # W depends on both U and X (proxy variable)
  W = numeric(n)
  for (i in 1:n) {
    # 使用归一化后的U和X计算logit，避免数值量级过大
    # Feature vector: [1, U_normalized[i], X_normalized[i,]]
    features = c(1, U_normalized[i], X_normalized[i,])
    logits_w = beta_w %*% features
    probs_w = exp(logits_w) / sum(exp(logits_w))
    W[i] = sample(1:w_dim, size=1, prob=probs_w)
  }
  
  # Generate T (A) using P(A|X) and P(A|U,X)
  # First compute P(A|X) using softmax
  # Note: beta_a was already generated above with fixed seed
  
  prop.x = numeric(n)
  epsilon = 1e-6  # 数值稳定性：防止prop.x接近0或1导致后续计算不稳定
  for (i in 1:n) {
    # 使用归一化后的X计算logit，避免数值量级过大
    x_vec = c(1, X_normalized[i,])
    logits_a = beta_a %*% x_vec
    probs_a = exp(logits_a) / sum(exp(logits_a))
    # 强制将概率限制在 [ε, 1-ε] 之间，避免数值不稳定
    prop.x[i] = max(epsilon, min(1 - epsilon, probs_a[2]))  # P(A=1|X=x)
  }
  
  # Compute P(A|U,X) using piecewise function
  # a(x) and b(x) from equations
  a.x = prop.x / (prop.x + Gamma * (1 - prop.x))
  b.x = prop.x / (prop.x + (1 - prop.x) / Gamma)
  
  # Compute p(x) and t(x) to ensure E[P(A|U,X)|X] = P(A|X)
  p.x = (1/(prop.x + (1-prop.x)/Gamma) - 1) / 
        (1/(prop.x + (1-prop.x)/Gamma) - 1/(prop.x + Gamma*(1-prop.x)))
  
  # 数值稳定性检查：由于prop.x已被限制在[ε, 1-ε]，理论上不应出现NaN/Inf
  # 但保留此检查作为双重保险
  if (any(is.na(p.x) | is.infinite(p.x))) {
    p.x[is.na(p.x) | is.infinite(p.x)] = prop.x[is.na(p.x) | is.infinite(p.x)]  # 使用prop.x作为fallback
  }
  p.x = pmax(pmin(p.x, 0.99), 0.01)  # 限制在[0.01, 0.99]范围内
  
  # For discrete U, we need to find threshold t(x) based on U distribution
  # Since U is discrete (values from 1 to u_dim), we use a quantile-based threshold
  # The threshold should be chosen such that approximately p(x) proportion of U values exceed it
  # For discrete case: we want P(|U - center| > t(x)) ≈ p(x)
  # We'll use quantiles of U levels based on p(x)
  u_center = (u_dim + 1) / 2  # Center of U levels (e.g., 3 for u_dim=5)
  
  # Compute P(A|U,X) for each observation
  # For discrete U, we use the indicator: 1{|U - center| > threshold}
  # The threshold is chosen to match p(x) approximately
  prop.xu = numeric(n)
  for (i in 1:n) {
    # Calculate threshold based on p(x): we want p(x) proportion in the tails
    # For discrete U, find the quantile threshold
    # If p(x) = 0.9, we want 90% of U values in the tails (wide tail region)
    # If p(x) = 0.1, we want 10% of U values in the tails (narrow tail region)
    # Lower threshold: floor(p.x[i]/2 * u_dim) - larger p.x gives larger threshold
    # Upper threshold: ceiling((1 - p.x[i]/2) * u_dim) - larger p.x gives smaller threshold
    lower_thresh = max(1, floor(p.x[i]/2 * u_dim))
    upper_thresh = min(u_dim, ceiling((1 - p.x[i]/2) * u_dim))
    
    # If U is in the tails (far from center), use a(x) (higher treatment probability)
    # Otherwise, use b(x) (lower treatment probability)
    if (U[i] <= lower_thresh || U[i] >= upper_thresh) {
      prop.xu[i] = a.x[i]
    } else {
      prop.xu[i] = b.x[i]
    }
  }
  
  # Generate T from P(A|U,X)
  TT = rbinom(n, size=1, prob=prop.xu)
  
  # Generate Y1 using softmax P(Y1|U,X)
  # Y1 does NOT depend on A (treatment), only on U and X
  # This is consistent with the continuous version: Y1 = X %*% beta + U
  Y1 = numeric(n)
  for (i in 1:n) {
    # 使用归一化后的U和X计算logit，避免数值量级过大
    # Feature vector: [1, U_normalized[i], X_normalized[i,]] (NO A/TT)
    features = c(1, U_normalized[i], X_normalized[i,])
    
    logits_y = beta_y %*% features
    probs_y = exp(logits_y) / sum(exp(logits_y))
    Y1[i] = sample(1:y_levels, size=1, prob=probs_y)
  }
  
  if (obs==FALSE){
    return(list("T"=TT, "X"=X, "U"=U, "W"=W, "Y1"=Y1, "ex"=prop.x, "exu"=prop.xu))
  }else{
    n_useful = sum(TT)
    while (n_useful < n){
      add.data = data.gen.ate(n, p, Gamma, beta, alpha0, FALSE, u_dim, w_dim, x_levels, y_levels, coeff_seed)
      X = rbind(X, add.data$X)
      U = c(U, add.data$U)  # U is now a vector
      W = c(W, add.data$W)  # W is now a vector
      Y1 = c(Y1, add.data$Y1)
      prop.x = c(prop.x, add.data$ex)
      prop.xu = c(prop.xu, add.data$exu)
      TT = c(TT, add.data$T)
      n_useful = sum(TT)
    }
    return(list("T"=TT, "X"=X, "U"=U, "W"=W, "Y1"=Y1, "ex"=prop.x, "exu"=prop.xu))
  }
}

############################################################
#####   output nonconformity score and trained model  ######
# X: covariate matrix
# Y: response (now discrete)
# train and output trained model if trained_model = NULL
# otherwise, use trained_model to get nonconformity score
# quantile is the coverage (1-alpha)
# Note: For discrete Y, we may need to adjust the scoring method
############################################################
conform.score <- function(X, Y, method='cqr', trained_model = NULL, quantile=0.9){
  if (method == 'cqr'){
    if (is.null(trained_model)){
      trained_model = quantile_forest(X, Y, num.threads = 2)
    }
    # fit quantiles
    qs = predict(trained_model, X, quantile=c((1-quantile)/2, 1-(1-quantile)/2))
    # ✅ 兼容不同 grf 版本的返回结构
    if (is.list(qs) && "predictions" %in% names(qs)) {
      qs <- qs$predictions
    }
    q_lo = qs[,1]
    q_hi = qs[,2]
    score = pmax( Y-q_hi, q_lo-Y )
  }
  return(list("model"=trained_model, "score"=score))
}


#################################### 
# compute WS-R lower bound for cdf  
# at a single point V[i]  
####################################
wsr.cdf.single <- function(delta, lx, ux, M, i, rand_given, rand_ind){
  n = length(lx)
  lxx = c(lx[1:i], rep(0,n-i))/M
  uxx = 1 + (c(rep(0,i),-ux[(i+1):n]) )/M
  if (rand_given){
    wsr1 = wsr_lower(delta/2, lxx[rand_ind])*M
    wsr2 = wsr_lower(delta/2, uxx[rand_ind])*M +1-M
  }else{
    wsr1 = wsr_lower(delta/2, sample(lxx))*M
    wsr2 = wsr_lower(delta/2, sample(uxx))*M +1-M
  }
  return( max(wsr1,wsr2))
}

#################################### 
# compute WS-R lower bound for cdf  
# at all points V[i]  
####################################

wsr.cdf <- function(delta, lx, ux, M, rand_given, rand_ind){
  n = length(lx)
  w_all <- sapply(1:n, FUN = wsr.cdf.single, delta=delta, lx=lx, ux=ux, 
                  M=M, rand_given = rand_given, rand_ind=rand_ind)
  return(w_all)
}

#################################### 
# find the hat{v} of pac procedure  
# for one single Gamma  
####################################

ate.single.pac <- function(Gamma, pp, calib.ex, calib.score, delta, alpha, rand_ind){
  # lower and upper bounds for likelihood ratio
  calib.lx = pp*(1+ (1-calib.ex)/(calib.ex*Gamma))
  calib.ux = pp*(1+ Gamma* (1-calib.ex)/(calib.ex))
  # aggregate information
  calib.all = data.frame("score"=calib.score, "lx"=calib.lx, "ux"=calib.ux, "ex" = calib.ex)
  # compute the 1-alpha quantile
  M = max(calib.ux)
  eta.gamma = wsr.qtl(delta, calib.all, M, alpha, rand_ind)
  return(eta.gamma)
}

############################################# 
# bi-search for quantile of wsr lower bound  
#############################################

wsr.qtl <- function(delta, calib.all, M, alpha, rand_ind){
  n_calib = nrow(calib.all)
  l.i = 1
  r.i = n_calib - 1
  m.i = floor((l.i+r.i)/2)
  left.cdf = wsr.cdf.single(delta, calib.all$lx, calib.all$ux, M, l.i, rand_given=TRUE, rand_ind)
  right.cdf = wsr.cdf.single(delta, calib.all$lx, calib.all$ux, M, r.i, rand_given=TRUE, rand_ind)
  if (right.cdf < 1-alpha){
    return(Inf)
  }
  mid.cdf = wsr.cdf.single(delta, calib.all$lx, calib.all$ux, M, m.i, rand_given=TRUE, rand_ind)
  gap = min(m.i - l.i, r.i - m.i)
  while( gap >0 ){
    if (mid.cdf < 1-alpha){
      l.i = m.i
      m.i = floor((l.i+r.i)/2)
      left.cdf = mid.cdf
      mid.cdf = wsr.cdf.single(delta, calib.all$lx, calib.all$ux, M, m.i, rand_given=TRUE, rand_ind)
    }else{
      r.i = m.i
      m.i = floor((l.i+r.i)/2)
      right.cdf = mid.cdf
      mid.cdf = wsr.cdf.single(delta, calib.all$lx, calib.all$ux, M, m.i, rand_given=TRUE, rand_ind)
    }
    gap = min(m.i - l.i, r.i - m.i)
    
  }
  if (mid.cdf >= 1-alpha){
    return(calib.all$score[m.i])
  }else{
    return(calib.all$score[r.i])
  }
}


######################################
# basic functions for WSR inequality  
######################################

compute_k_lower <- function(x, mu, nu){
  kterms = 1 + nu * (x - mu)
  ks = rep(kterms[1],length(kterms))
  for (ii in 2:length(ks)){
    ks[ii] = ks[ii-1]*kterms[ii]
  }
  return(max(ks))
}

compute_k_upper <- function(x, mu, nu){
  kterms = 1 - nu * (x - mu)
  ks = rep(kterms[1],length(kterms))
  for (ii in 2:length(ks)){
    ks[ii] = ks[ii-1]*kterms[ii]
  }
  return(max(ks))
}

wsr_lower <- function(delta, x){
  n = length(x)
  mu_hat <- (1/2 + cumsum(x)) / (1 : n + 1)
  sig_hat = (1/4 + cumsum((x-mu_hat)^2))/(1:n +1)
  # nu[i] = xxxx/simga_{i-1}
  nu <- pmin(1, sqrt(2 * log(1 / delta) / (n * sig_hat^2)))
  nu[2: length(nu)] = nu[1:(length(nu)-1)]
  nu[1] = pmin(1, sqrt(2 * log(1 / delta) / (n /4)))
  u_list <- (1 : 1000) / 1000 
  k_all <- sapply(u_list, FUN = compute_k_lower, x = x, nu = nu)
  u_ind <- min(which(k_all <= 1 / delta))
  if (u_ind == Inf){
    u_ind = 1000
  }
  bnd <- u_list[u_ind]
  return(bnd)
}

wsr_upper <- function(delta, x){
  n = length(x)
  mu_hat <- (1/2 + cumsum(x)) / (1 : n + 1)
  sig_hat = (1/4 + cumsum((x-mu_hat)^2))/(1:n +1)
  nu <- pmin(1, sqrt(2 * log(1 / delta) / (n * sig_hat^2)))
  u_list <- (1 : 1000) / 1000 
  k_all <- sapply(u_list, FUN = compute_k_upper, x = x, nu = nu)
  u_ind <- min(which(k_all > 1 / delta))
  if (u_ind == Inf){
    u_ind = 1000
  }
  bnd <- u_list[u_ind]
  return(bnd)
}

