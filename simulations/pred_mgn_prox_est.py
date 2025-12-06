#!/usr/bin/env python3
"""
Proximal method with estimated bounds
Converted from pred_mgn_prox_est.R
"""

import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path to import utils_new
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils_new.util_ate_multi import data_gen_ate, conform_score
from utils_new.neural_models import QuantileForestNN, RegressionForestNN


def compute_lx_ux_from_e(e_model, X_mat, W_grid, pbar, eps=1e-3):
    """
    Compute proximal bounds using estimated e(X,W) model
    
    Args:
        e_model: Trained RegressionForestNN model for e(X,W)
        X_mat: [n, p] covariate matrix
        W_grid: [K, w_dim] W grid
        pbar: Population P(T=1)
        eps: Epsilon for clipping
    
    Returns:
        Dictionary with 'lx' and 'ux' arrays
    """
    n = X_mat.shape[0]
    K = W_grid.shape[0]
    
    if n < 1 or K < 1:
        raise ValueError("Empty X_mat or W_grid in compute_lx_ux_from_e")
    
    # Repeat X for each W in grid
    X_rep = np.repeat(X_mat, K, axis=0)  # [n*K, p]
    W_rep = np.tile(W_grid, (n, 1))  # [n*K, w_dim]
    XW = np.hstack([X_rep, W_rep])  # [n*K, p+w_dim]
    
    # Predict e(X,W) for all combinations
    ex_pred = e_model.predict(XW)['predictions']
    ex_mat = ex_pred.reshape(n, K)  # [n, K]
    
    # Clip and compute min/max
    ex_mat = np.clip(ex_mat, eps, 1 - eps)
    emax = np.max(ex_mat, axis=1)  # [n]
    emin = np.min(ex_mat, axis=1)  # [n]
    
    lx = pbar / emax
    ux = pbar / emin
    
    return {'lx': lx, 'ux': ux}


def main():
    # Parse command line arguments
    if len(sys.argv) < 7:
        print("Usage: python pred_mgn_prox_est.py <p> <n> <w_dim> <alpha_ind> <Gamma_ind> <seed>")
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
    
    print(f" - Running pred_mgn_prox_est: estimated bounds (robust + unaware + proximal), "
          f"alpha {alpha}, Gamma {Gamma}, n {n}, p {p}, w_dim {w_dim}, seed {seed}")
    
    # Output directory (estimated bounds version, separated by w_dim)
    base_out_dir = "/projects/p32685/cfsensitivity_results/simulation_prox_est/"
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
    
    # Estimate hat{p} and e(x) (estimated bounds version)
    hat_p = np.mean(train_data['T'])
    e_model = RegressionForestNN(train_data['X'], train_data['T'], device='cuda')
    
    # Proximal: Train proxy-based propensity (also based on observed data)
    # Generate proxy W for all training data
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
    
    train_XW_full = np.hstack([train_data['X'], train_W_full])
    e_model_prox = RegressionForestNN(train_XW_full, train_data['T'], device='cuda')
    
    # Sample candidate W grid from training W (for robustness and efficiency)
    np.random.seed(seed + 123)
    w_grid_size = min(50, train_W_full.shape[0])
    W_grid = train_W_full[np.random.choice(train_W_full.shape[0], w_grid_size, replace=False), :]
    
    # Calibration
    calib_data = data_gen_ate(n, p, Gamma, beta, alpha0, obs=True, u_dim=u_dim)
    calib_T_indices = np.where(calib_data['T'] == 1)[0][:n]
    calib_X = calib_data['X'][calib_T_indices]
    calib_Y = calib_data['Y1'][calib_T_indices]
    calib_U = calib_data['U'][calib_T_indices]  # U is matrix
    n_calib = len(calib_Y)
    
    # Estimate e(x) for calibration set
    calib_ex = e_model.predict(calib_X)['predictions']
    
    # Lower and upper bounds of weight function (estimated version uses hat.p)
    calib_lx = hat_p * (1 + (1 - calib_ex) / (calib_ex * Gamma))
    calib_ux = hat_p * (1 + Gamma * (1 - calib_ex) / (calib_ex))
    calib_nc_w = hat_p / calib_ex
    
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
    
    # Estimate e(x) for test set
    test_ex = e_model.predict(test_X)['predictions']
    test_lx = hat_p * (1 + 1 / Gamma * (1 - test_ex) / test_ex)
    test_ux = hat_p * (1 + Gamma * (1 - test_ex) / test_ex)
    
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
    nc_test_weight = hat_p / test_ex
    
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
    
    # Step 1: Estimate proxy-based propensity score (for constructing bounds)
    # e.model.prox already trained in training phase
    # Step 2: Compute proximal bounds for calibration and test sets (does not depend on Gamma)
    prox_bounds_calib = compute_lx_ux_from_e(e_model_prox, calib_X, W_grid, pbar=hat_p)
    calib_lx_prox = prox_bounds_calib['lx']
    calib_ux_prox = prox_bounds_calib['ux']
    
    prox_bounds_test = compute_lx_ux_from_e(e_model_prox, test_X, W_grid, pbar=hat_p)
    test_lx_prox = prox_bounds_test['lx']
    test_ux_prox = prox_bounds_test['ux']
    
    # Step 3: Use same weighted conformal procedure as confounding-aware (only replace l/u)
    prox_sum_num = np.zeros(n_calib)  # for numerator
    prox_sum_den = np.zeros(n_calib)  # for denominator
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
    
    # Evaluate coverage (consistent with original procedure)
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
        'method': ['marginal_plus_proximal_est']
    })
    
    save_path = os.path.join(out_dir,
                            f"pred_marginal_prox_p_{p}_n_{n}_wdim_{w_dim}_alpha_{alpha_ind}_gamma_{Gamma_ind}_seed_{seed}.csv")
    res.to_csv(save_path, index=False)
    
    print(f" - Results saved. Coverages: robust={np.mean(c_cover):.3f}, "
          f"unaware={np.mean(nc_cover):.3f}, proximal={np.mean(prox_cover):.3f}")


if __name__ == "__main__":
    main()

