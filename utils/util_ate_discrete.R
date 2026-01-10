############################################################
#####   discrete data generating process (Binomial)   ######
# n: output sample size 
# p: covariate dimension 
# Gamma: confounding level
# beta, alpha0: linear coefficients (for compatibility)
# u_dim: must be EVEN (default 20), determines range of centered U (n_trials = u_dim - 1 internally)
# w_dim: number of discrete categories for W (default 20, W takes values {1, 2, ..., w_dim})
# x_levels: number of discrete categories for X (default 10, X takes values {1, 2, ..., x_levels})
# y_levels: number of discrete categories for Y1 (default 100, Y1 takes values {1, 2, ..., y_levels})
# obs = TRUE generate observations (training data) of size n
# obs = FALSE generate all data of size n
# U generation: Binomial(u_dim-1, p_u(X)) then centered to range [-(u_dim-1)/2, (u_dim-1)/2]
#               where p_u(X) = logistic([1, X] %*% beta_u_vec)
# T generation: randomized threshold method to ensure E[P(A|U,X)|X] = P(A|X) exactly
############################################################

data.gen.ate <- function(n, p, Gamma, beta, alpha0=0, obs=TRUE, u_dim=20, w_dim=20, x_levels=10, y_levels=100, coeff_seed=12345){
  # Check that u_dim is even
  if (u_dim %% 2 != 0) {
    stop("u_dim must be even for proper centering")
  }
  
  # Set n_trials based on u_dim (ensures U has u_dim possible values)
  n_trials = u_dim - 1
  
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
  
  # Initialize coefficient vectors with dynamic intercept strategy
  # Strategy: Keep slopes moderate to avoid sigmoid saturation, and compute
  # intercept dynamically to center p_u(X) around 0.5, ensuring U can take
  # all possible values symmetrically. This avoids the polarization effect.
  beta_slopes = runif(p, -0.8, 0.8)  # Moderate slopes to avoid saturation
  # Note: intercept will be computed dynamically after X is normalized
  
  # For P(W|U,X): beta_w is (w_dim x (p+2)) matrix
  # Features: [1 (intercept), U (continuous centered), X (p)] = 1 + 1 + p = p + 2
  # Note: β_{w,U} must be non-zero to ensure W is relevant to U
  # 【关键修改】增强 W 对 U 的依赖：分别生成不同范围的列，然后组合
  # 这样既保持随机数序列一致（确保 robust/unaware 结果不变），又让 U 列独立从更大范围采样
  # 第1列（intercept）：[-0.5, 0.5]
  beta_w_col1 = runif(w_dim, -0.5, 0.5)
  # 第2列（U列）：[-1.5, 1.5] - 增强依赖，使 W 携带更多关于 U 的信息
  beta_w_U = runif(w_dim, -1.5, 1.5)
  # 第3到p+2列（X列）：[-0.5, 0.5]
  beta_w_X = matrix(runif(w_dim*p, -0.5, 0.5), nrow=w_dim, ncol=p)
  # 组合成完整矩阵
  beta_w = cbind(beta_w_col1, beta_w_U, beta_w_X)
  
  # For P(Y1|U,X): beta_y is (y_levels x (p+2)) matrix
  # Features: [1 (intercept), U (continuous centered), X (p)] = 1 + 1 + p = p + 2
  # Note: Y1 does NOT depend on A (treatment), only on U and X (for counterfactual prediction)
  beta_y = matrix(runif(y_levels*(p+2), -0.5, 0.5), nrow=y_levels, ncol=p+2)
  
  # For P(A|X): beta_a is (2 x (p+1)) matrix
  # Features: [1 (intercept), X (p)] = p + 1
  beta_a = matrix(runif(2*(p+1), -0.5, 0.5), nrow=2, ncol=p+1)
  
  # Restore random seed (if it existed before)
  if (seed_exists) {
    assign(".Random.seed", old_seed, envir = .GlobalEnv)
  }
  
  # 归一化X，将类别索引映射到[0,1]
  # 将 1~x_levels 映射到 0~1
  X_normalized = (X - 1) / (x_levels - 1)
  
  # Generate U using Center-Shifted Binomial with dynamic intercept
  # Strategy: Compute intercept dynamically to center p_u(X) around 0.5
  # This ensures U can take all possible values symmetrically
  # Step 1: Compute linear part (slopes only, no intercept)
  linear_part = X_normalized %*% beta_slopes  # n x 1 vector
  
  # Step 2: Compute dynamic intercept to center linear_pred at 0
  # This makes p_u(X) centered around 0.5, ensuring symmetric U distribution
  intercept = -mean(linear_part)
  
  # Step 3: Compute p_u(X) = logistic(intercept + linear_part)
  linear_pred = intercept + linear_part  # n x 1 vector
  p_u = plogis(linear_pred)  # n x 1 vector
  
  # Step 4: Sample U_raw ~ Binomial(n_trials, p_u(X)) and center
  U_raw = rbinom(n, size=n_trials, prob=p_u)
  U = U_raw - n_trials/2
  
  # Debugging: Check U generation (disabled - already verified)
  # if (obs == FALSE) {
  #   ... (debugging code removed)
  # }
  
  # 归一化U用于后续 W 和 Y1 的生成（映射到合理的数值范围）
  # U 范围: [-n_trials/2, n_trials/2], 归一化到 [-1, 1]
  U_normalized = U / (n_trials/2)
  
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
  
  # Randomized threshold method for discrete U (Binomial)
  # Goal: Ensure P(Tail | X=x) = p(x) EXACTLY through randomization
  # Key: U is now centered around 0, ranging from -(n_trials/2) to +(n_trials/2)
  
  prop.xu = numeric(n)
  for (i in 1:n) {
    # Step 1: Compute P(U|X=x_i) for all possible U values
    # For Binomial: U_raw ~ Binomial(n_trials, p_u[i]), then U = U_raw - n_trials/2
    # So P(U = k | X=x_i) = P(U_raw = k + n_trials/2 | X=x_i) = dbinom(k + n_trials/2, n_trials, p_u[i])
    
    # All possible U values: -(n_trials/2) to +(n_trials/2)
    u_values = seq(-n_trials/2, n_trials/2, by=1)
    u_raw_values = u_values + n_trials/2  # Convert back to U_raw scale [0, n_trials]
    
    # Compute probabilities for all possible U values
    probs_u = dbinom(u_raw_values, size=n_trials, prob=p_u[i])
    
    # Step 2: Compute |U| (distance from center = 0)
    u_abs = abs(u_values)
    
    # Step 3: Sort by |U| from large to small (tails first)
    sorted_indices = order(u_abs, decreasing=TRUE)
    sorted_u_abs = u_abs[sorted_indices]
    sorted_probs = probs_u[sorted_indices]
    sorted_u_values = u_values[sorted_indices]
    
    # Step 4: Find threshold k* such that:
    # sum_{|u| > k*} P(U=u|X) < p(x) <= sum_{|u| >= k*} P(U=u|X)
    cumprob = 0
    threshold_index = 1  # Default: all in tail
    threshold_distance = sorted_u_abs[1]
    
    for (j in 1:length(sorted_indices)) {
      if (cumprob >= p.x[i]) {
        # Found the threshold: previous distance was k*
        threshold_index = j - 1
        if (threshold_index >= 1) {
          threshold_distance = sorted_u_abs[threshold_index]
        }
        break
      }
      cumprob = cumprob + sorted_probs[j]
      threshold_distance = sorted_u_abs[j]
      threshold_index = j
    }
    
    # Step 5: Compute fill probability r
    # cumprob_before_k_star = sum of probabilities for |u| > threshold_distance
    # cumprob_at_k_star = probability at |u| = threshold_distance
    cumprob_before = 0
    prob_at_threshold = 0
    
    for (j in 1:length(sorted_indices)) {
      if (sorted_u_abs[j] > threshold_distance) {
        cumprob_before = cumprob_before + sorted_probs[j]
      } else if (sorted_u_abs[j] == threshold_distance) {
        prob_at_threshold = prob_at_threshold + sorted_probs[j]
      }
    }
    
    # r = (p(x) - cumprob_before) / prob_at_threshold
    if (prob_at_threshold > 1e-10) {
      r = (p.x[i] - cumprob_before) / prob_at_threshold
      r = pmax(0, pmin(1, r))  # Clip to [0, 1]
    } else {
      r = 0
    }
    
    # Step 6: Determine if U[i] is in Tail
    u_i_abs = abs(U[i])
    
    if (u_i_abs > threshold_distance) {
      # Definitely in Tail
      is_tail = TRUE
    } else if (u_i_abs < threshold_distance) {
      # Definitely in Center
      is_tail = FALSE
    } else {
      # At boundary: randomize with probability r
      is_tail = (runif(1) < r)
    }
    
    # Step 7: Set prop.xu based on Tail/Center status
    if (is_tail) {
      prop.xu[i] = a.x[i]  # Tail: use lower bound (less likely to be treated)
    } else {
      prop.xu[i] = b.x[i]  # Center: use upper bound (more likely to be treated)
    }
  }
  
  # Generate T from P(A|U,X)
  TT = rbinom(n, size=1, prob=prop.xu)
  
  # Debugging: Check T generation (disabled - already verified)
  # if (obs == FALSE) {
  #   ... (debugging code removed)
  # }
  
  # Generate Y1 using direct linear combination (no normalization)
  # Y1 does NOT depend on A (treatment), only on U and X
  # Method: Y_latent = base_intercept + X_effect + U_effect, then discretize
  # This gives clear physical meaning: gamma_u = 1.0 means 1 integer unit offset
  
  # Use continuous version's beta coefficients
  beta_continuous = c(-0.531, 0.126, -0.312, 0.018, rep(0, p-4)) * 2.0
  beta_matrix = matrix(beta_continuous, nrow=p)
  
  # X signal scaling factor (controls X's contribution to Y)
  # X_normalized is in [0,1], beta coefficients are small, so we need to scale up
  X_scale = 12  # Adjustable: controls how much X can affect Y (in integer units)
  X_effect = as.vector(X_normalized %*% beta_matrix) * X_scale
  
  # U scaling factor (controls confounding strength in Y generation)
  # U is in [-(n_trials/2), (n_trials/2)] = [-9.5, 9.5] for u_dim=20
  # gamma_u = 1.0 means U can offset Y by up to 9.5 integer units
  gamma_u = 1.5  # Adjustable parameter: direct integer unit offset
  
  # U's contribution to Y
  U_effect = gamma_u * U
  
  # Generate latent continuous Y using direct linear combination
  # Base intercept centers Y around y_levels/2 to ensure reasonable range
  base_intercept = y_levels / 2  # 50 for y_levels=100
  Y_latent = base_intercept + X_effect + U_effect
  
  # Direct discretization (round to nearest integer)
  Y1 = round(Y_latent)
  
  # Ensure Y1 is within bounds (safety check)
  Y1 = pmax(1, pmin(y_levels, Y1))
  
  if (obs==FALSE){
    return(list("T"=TT, "X"=X, "U"=U, "W"=W, "Y1"=Y1, "ex"=prop.x, "exu"=prop.xu))
  }else{
    n_useful = sum(TT)
    while (n_useful < n){
      add.data = data.gen.ate(n, p, Gamma, beta, alpha0, FALSE, u_dim, w_dim, x_levels, y_levels, coeff_seed)
      X = rbind(X, add.data$X)
      U = c(U, add.data$U)  # U is now a vector (centered)
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
conform.score <- function(X, Y, method='cqr', trained_model = NULL, quantile=0.9, Y_original = NULL){
  # Y_original: 如果提供，用于计算score（当Y是添加噪声后的连续版本时）
  # 这样可以让RF用连续Y学习分位数，但用原始离散Y计算score
  if (method == 'cqr'){
    if (is.null(trained_model)){
      # 调整参数以更好地处理离散Y
      # 对离散Y添加小噪声后，RF能更好地学习分位数
      trained_model = quantile_forest(X, Y, 
                                      num.trees = 2000,  # 增加树的数量
                                      min.node.size = 5,  # 节点大小
                                      num.threads = 1)
    }
    # fit quantiles
    qs = predict(trained_model, X, quantile=c((1-quantile)/2, 1-(1-quantile)/2))
    # ✅ 兼容不同 grf 版本的返回结构
    if (is.list(qs) && "predictions" %in% names(qs)) {
      qs <- qs$predictions
    }
    q_lo = qs[,1]
    q_hi = qs[,2]
    # 如果提供了Y_original，用原始Y计算score；否则用Y计算
    Y_for_score = if (!is.null(Y_original)) Y_original else Y
    score = pmax( Y_for_score-q_hi, q_lo-Y_for_score )
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

