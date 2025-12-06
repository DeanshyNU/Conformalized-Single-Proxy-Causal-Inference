#!/usr/bin/env python3
"""
Proximal method with true bounds
Converted from pred_mgn_prox.R
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import norm

# Add parent directory to path to import utils_new
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils_new.util_ate_multi import data_gen_ate, conform_score
from utils_new.neural_models import QuantileForestNN


def compute_lx_ux_from_e_true(X_mat, W_grid, pbar, Gamma, beta, alpha0, noise_level, eps=1e-3):
    """
    Compute true e(X,W) = P(T=1|X,W) using closed-form formula
    Fully vectorized version without for-loops
    
    Args:
        X_mat: [n, p] covariate matrix
        W_grid: [K, w_dim] W grid
        pbar: Population P(T=1)
        Gamma: Confounding level
        beta: [p, 1] linear coefficients
        alpha0: Intercept
        noise_level: Noise level for W
        eps: Epsilon for clipping
    
    Returns:
        Dictionary with 'lx' and 'ux' arrays
    """
    n = X_mat.shape[0]
    K = W_grid.shape[0]
    
    if n < 1 or K < 1:
        raise ValueError("Empty X_mat or W_grid in compute_lx_ux_from_e_true")
    
    # Extract first column of W_grid (corresponding to U1)
    W1_grid = W_grid[:, 0]  # K × 1 vector
    
    # Step 1: Compute basic quantities for all x (vectorized, n × 1)
    logit_ex = alpha0 + X_mat @ beta
    ex = (np.exp(logit_ex) / (1 + np.exp(logit_ex))).flatten()  # Ensure 1D
    
    ax = (ex / (ex + Gamma * (1 - ex))).flatten()  # Ensure 1D
    bx = (ex / (ex + (1 - ex) / Gamma)).flatten()  # Ensure 1D
    
    s_x = np.abs(1 + 0.5 * np.sin(2.5 * X_mat[:, 0]))
    
    prop_x = ex
    p_x = ((1 / (prop_x + (1 - prop_x) / Gamma) - 1) / \
          (1 / (prop_x + (1 - prop_x) / Gamma) - 1 / (prop_x + Gamma * (1 - prop_x)))).flatten()
    t_x = (norm.ppf(1 - p_x / 2) * s_x).flatten()  # Ensure 1D
    
    # Step 2: Compute posterior distribution parameters (vectorized, n × 1)
    sigma_w_sq = noise_level**2
    tau_sq = (1 / (1 / sigma_w_sq + 1 / (s_x**2))).flatten()  # Ensure 1D
    tau = np.sqrt(tau_sq)
    
    # Step 3: Use matrix broadcasting to compute μ(x,w) for all (x, w) combinations
    # tau_sq is n × 1, W1_grid is K × 1
    # Use outer product to get n × K matrix: mu_mat[i,k] = tau_sq[i] * W1_grid[k] / sigma_w_sq
    mu_mat = np.outer(tau_sq / sigma_w_sq, W1_grid)  # n × K
    
    # Step 4: Compute z_upper and z_lower (matrix broadcasting)
    # t_x is n × 1, tau is n × 1, mu_mat is n × K
    # Need to expand t_x and tau to n × K
    t_x_mat = np.tile(t_x.reshape(-1, 1), (1, K))  # n × K (each column same)
    tau_mat = np.tile(tau.reshape(-1, 1), (1, K))  # n × K (each column same)
    
    z_upper_mat = (t_x_mat - mu_mat) / tau_mat  # n × K
    z_lower_mat = (-t_x_mat - mu_mat) / tau_mat  # n × K
    
    # Step 5: Compute q(x,w) matrix (n × K)
    q_mat = 1 - norm.cdf(z_upper_mat) + norm.cdf(z_lower_mat)
    
    # Step 6: Compute e(x,w) matrix (n × K)
    # ax and bx are n × 1, need to expand to n × K
    ax_mat = np.tile(ax.reshape(-1, 1), (1, K))  # n × K (each column same)
    bx_mat = np.tile(bx.reshape(-1, 1), (1, K))  # n × K (each column same)
    ex_mat = ax_mat * q_mat + bx_mat * (1 - q_mat)
    
    # Step 7: Clip and take max/min of each row
    ex_mat = np.clip(ex_mat, eps, 1 - eps)
    emax = np.max(ex_mat, axis=1)  # n × 1
    emin = np.min(ex_mat, axis=1)  # n × 1
    
    lx = pbar / emax
    ux = pbar / emin
    
    return {'lx': lx, 'ux': ux}


def main():
    # Parse command line arguments
    if len(sys.argv) < 7:
        print("Usage: python pred_mgn_prox.py <p> <n> <w_dim> <alpha_ind> <Gamma_ind> <seed>")
        sys.exit(1)
    
    p = int(sys.argv[1])
    n = int(sys.argv[2])
    w_dim = int(sys.argv[3])
    alpha_ind = int(sys.argv[4])
    Gamma_ind = int(sys.argv[5])
    seed = int(sys.argv[6])
    
    alphas = np.arange(0.1, 1.0, 0.1)
    gammas = np.array([1.5, 2, 2.5, 3, 5])
    
    # Coverage target 1-alpha
    alpha = alphas[alpha_ind - 1]  # R uses 1-indexed
    # Confounding level Gamma
    Gamma = gammas[Gamma_ind - 1]  # R uses 1-indexed
    
    print(f" - Running pred_mgn_prox: robust + unaware + proximal methods, alpha {alpha}, "
          f"Gamma {Gamma}, n {n}, p {p}, w_dim {w_dim}, seed {seed}")
    
    # Output directory (true bounds version, separated by w_dim)
    base_out_dir = "/projects/p32685/cfsensitivity_results/simulation_prox_true/"
    out_dir = os.path.join(base_out_dir, f"wdim_{w_dim}")
    os.makedirs(out_dir, exist_ok=True)
    
    # Parameters
    alpha0 = 0
    n_test = 500
    u_dim = 20  # U fixed at 20 dimensions
    beta_base = np.array([[-0.531], [0.126], [-0.312], [0.018]])
    if p > 4:
        beta = np.vstack([beta_base, np.zeros((p - 4, 1))])
    else:
        beta = beta_base[:p]
    
    # Generate true probability of treatment
    np.random.seed(seed)
    temp_data = data_gen_ate(n * 1000, p, Gamma, beta, alpha0, obs=False, u_dim=u_dim)
    pp = np.mean(temp_data['T'])
    
    # Set seed
    np.random.seed(seed)
    
    # Fit on the training fold
    # Generate data with proxy W (U is 20-dimensional matrix)
    train_data = data_gen_ate(n, p, Gamma, beta, alpha0, obs=True, u_dim=u_dim)
    
    # Only use treated samples for CQR training, consistent with original method
    train_T_indices = np.where(train_data['T'] == 1)[0][:n]
    train_X = train_data['X'][train_T_indices]
    train_Y = train_data['Y1'][train_T_indices]
    train_U = train_data['U'][train_T_indices]  # U is now n × 20 matrix
    
    # Proximal: Generate proxy W of different dimensions based on w_dim
    noise_level = 0.5
    # When w_dim > u_dim, create fixed mixing matrix (for reproducibility)
    if w_dim > u_dim:
        np.random.seed(seed + 999)  # Fixed seed for reproducibility
        extra_cols = w_dim - u_dim
        M_extra = np.random.normal(0, 1, size=(u_dim, extra_cols))
    else:
        M_extra = None
    
    # Train the nonconformity score function (original method)
    train_score = conform_score(train_X, train_Y, method='cqr', trained_model=None,
                                quantile=1 - alpha, device='cuda')
    t_mdl = train_score['model']
    
    # Calibration
    calib_data = data_gen_ate(n, p, Gamma, beta, alpha0, obs=True, u_dim=u_dim)
    calib_T_indices = np.where(calib_data['T'] == 1)[0][:n]
    calib_X = calib_data['X'][calib_T_indices]
    calib_Y = calib_data['Y1'][calib_T_indices]
    calib_U = calib_data['U'][calib_T_indices]  # U is matrix
    calib_ex = calib_data['ex'][calib_T_indices]
    n_calib = len(calib_Y)
    
    # Lower and upper bounds of weight function (original method)
    calib_lx = pp * (1 + (1 - calib_ex) / (calib_ex * Gamma))
    calib_ux = pp * (1 + Gamma * (1 - calib_ex) / (calib_ex))
    calib_nc_w = pp / calib_data['ex'][calib_T_indices]
    
    # Non-conformity score on calibration data (original method)
    calib_score = conform_score(calib_X, calib_Y, method='cqr', trained_model=t_mdl,
                                quantile=1 - alpha, device='cuda')['score']
    
    # Create DataFrame and sort by score
    calib_all = pd.DataFrame({
        'score': calib_score,
        'lx': calib_lx,
        'ux': calib_ux,
        'wx': calib_nc_w,
        'ex': calib_ex
    })
    calib_all = calib_all.sort_values('score').reset_index(drop=True)
    
    # Generate test fold
    test_data = data_gen_ate(n_test, p, Gamma, beta, alpha0, obs=False, u_dim=u_dim)
    test_X = test_data['X']
    test_Y1 = test_data['Y1']
    test_U = test_data['U']  # U is matrix
    test_ex = test_data['ex']
    test_lx = pp * (1 + 1 / Gamma * (1 - test_ex) / test_ex)
    test_ux = pp * (1 + Gamma * (1 - test_ex) / test_ex)
    
    # Predictions (original method)
    test_pred = t_mdl.predict(test_X, quantile=[alpha / 2, 1 - alpha / 2])
    if isinstance(test_pred, dict) and 'predictions' in test_pred:
        test_pred = test_pred['predictions']
    
    # Method 1: confounding-aware algorithm (original)
    print(" - Computing the robust weighted conformal inference...")
    
    # Partial sums for confounding-aware
    sum_num = np.zeros(n_calib)  # for numerator
    sum_den = np.zeros(n_calib)  # for denominator
    sum_num[0] = calib_all['lx'].iloc[0]
    sum_den[0] = calib_all['lx'].iloc[0] + np.sum(calib_all['ux'].iloc[1:])
    
    for k in range(1, n_calib):
        sum_num[k] = sum_num[k - 1] + calib_all['lx'].iloc[k]
        sum_den[k] = sum_den[k - 1] - calib_all['ux'].iloc[k] + calib_all['lx'].iloc[k]
    
    # Confounding-aware prediction
    c_test_lo = np.zeros(n_test)
    c_test_hi = np.zeros(n_test)
    
    for ii in range(n_test):
        ratios = sum_num / (sum_den + test_ux[ii])
        kstar = np.where(ratios > 1 - alpha)[0]
        if len(kstar) > 0:
            kstar = kstar[0]
            v_kstar = calib_all['score'].iloc[kstar]
            c_test_lo[ii] = test_pred[ii, 0] - v_kstar
            c_test_hi[ii] = test_pred[ii, 1] + v_kstar
    
    # Evaluate coverage on test data
    c_cover = (c_test_lo <= test_Y1) * (c_test_hi >= test_Y1)
    
    print("Done.")
    
    # Method 2: confounding-unaware algorithm (original)
    print(" - Computing the vanilla weighted conformal inference...")
    
    nc_sum = np.zeros(n_calib)
    nc_sum[0] = calib_all['wx'].iloc[0]
    for k in range(1, n_calib):
        nc_sum[k] = nc_sum[k - 1] + calib_all['wx'].iloc[k]
    
    nc_test_lo = np.zeros(n_test)
    nc_test_hi = np.zeros(n_test)
    nc_test_weight = pp / test_ex
    
    for ii in range(n_test):
        nc_ratios = nc_sum / (nc_sum[n_calib - 1] + nc_test_weight[ii])
        nc_kstar = np.where(nc_ratios > 1 - alpha)[0]
        if len(nc_kstar) > 0:
            nc_kstar = nc_kstar[0]
            nc_v_kstar = calib_all['score'].iloc[nc_kstar]
            nc_test_lo[ii] = test_pred[ii, 0] - nc_v_kstar
            nc_test_hi[ii] = test_pred[ii, 1] + nc_v_kstar
    
    nc_cover = (nc_test_lo <= test_Y1) * (nc_test_hi >= test_Y1)
    
    print("Done.")
    
    # Method 3: Proximal CSPCI method
    print(" - Computing the proximal conformal inference...")
    
    # Sample candidate W grid from training W (for robustness and efficiency)
    # Generate proxy W for all training data (for sampling W grid)
    n_train_total = train_data['X'].shape[0]
    if w_dim < u_dim:
        train_W_full = train_data['U'][:, :w_dim] + \
                      np.random.normal(0, noise_level, size=(n_train_total, w_dim))
    elif w_dim == u_dim:
        train_W_full = train_data['U'] + \
                      np.random.normal(0, noise_level, size=(n_train_total, u_dim))
    else:
        extra_cols = w_dim - u_dim
        W_extra_full = train_data['U'] @ M_extra + \
                      np.random.normal(0, noise_level, size=(n_train_total, extra_cols))
        train_W_full = np.hstack([
            train_data['U'] + np.random.normal(0, noise_level, size=(n_train_total, u_dim)),
            W_extra_full
        ])
    
    np.random.seed(seed + 123)
    w_grid_size = min(50, train_W_full.shape[0])
    W_grid = train_W_full[np.random.choice(train_W_full.shape[0], w_grid_size, replace=False), :]
    
    # Step 2: Compute true proxy-based bounds for calibration set
    prox_bounds_calib = compute_lx_ux_from_e_true(calib_X, W_grid, pbar=pp, Gamma=Gamma,
                                                  beta=beta, alpha0=alpha0,
                                                  noise_level=noise_level)
    calib_lx_prox = prox_bounds_calib['lx']
    calib_ux_prox = prox_bounds_calib['ux']
    
    # Step 3: Compute true proximal bounds for test set
    prox_bounds_test = compute_lx_ux_from_e_true(test_X, W_grid, pbar=pp, Gamma=Gamma,
                                                 beta=beta, alpha0=alpha0,
                                                 noise_level=noise_level)
    test_lx_prox = prox_bounds_test['lx']
    test_ux_prox = prox_bounds_test['ux']
    
    # Step 4: Use same weighted conformal procedure as confounding-aware (only replace l/u)
    prox_sum_num = np.zeros(n_calib)
    prox_sum_den = np.zeros(n_calib)
    prox_sum_num[0] = calib_lx_prox[0]
    prox_sum_den[0] = calib_lx_prox[0] + np.sum(calib_ux_prox[1:])
    
    for k in range(1, n_calib):
        prox_sum_num[k] = prox_sum_num[k - 1] + calib_lx_prox[k]
        prox_sum_den[k] = prox_sum_den[k - 1] - calib_ux_prox[k] + calib_lx_prox[k]
    
    prox_test_lo = np.zeros(n_test)
    prox_test_hi = np.zeros(n_test)
    
    for ii in range(n_test):
        prox_ratios = prox_sum_num / (prox_sum_den + test_ux_prox[ii])
        prox_kstar = np.where(prox_ratios > 1 - alpha)[0]
        if len(prox_kstar) > 0:
            prox_kstar = prox_kstar[0]
            prox_v_kstar = calib_all['score'].iloc[prox_kstar]
            prox_test_lo[ii] = test_pred[ii, 0] - prox_v_kstar
            prox_test_hi[ii] = test_pred[ii, 1] + prox_v_kstar
    
    # Evaluate coverage
    prox_cover = (prox_test_lo <= test_Y1) * (prox_test_hi >= test_Y1)
    
    print("Done.")
    
    # Output summary of test (output results for three methods)
    res = pd.DataFrame({
        # Original two methods
        'c.cov': [np.mean(c_cover)],
        'c.len': [np.mean(c_test_hi - c_test_lo)],
        'nc.cov': [np.mean(nc_cover)],
        'nc.len': [np.mean(nc_test_hi - nc_test_lo)],
        # Proximal method
        'prox.cov': [np.mean(prox_cover)],
        'prox.len': [np.mean(prox_test_hi - prox_test_lo)],
        # Other information
        'n': [n],
        'p': [p],
        'u_dim': [u_dim],
        'w_dim': [w_dim],
        'n_calib': [n_calib],
        'gamma': [Gamma],
        'alpha': [alpha],
        'seed': [seed],
        'method': ['marginal_plus_proximal']
    })
    
    save_path = os.path.join(out_dir,
                            f"pred_marginal_prox_p_{p}_n_{n}_wdim_{w_dim}_alpha_{alpha_ind}_gamma_{Gamma_ind}_seed_{seed}.csv")
    res.to_csv(save_path, index=False)
    
    print(f" - Results saved. Coverages: robust={np.mean(c_cover):.3f}, "
          f"unaware={np.mean(nc_cover):.3f}, proximal={np.mean(prox_cover):.3f}")


if __name__ == "__main__":
    main()

