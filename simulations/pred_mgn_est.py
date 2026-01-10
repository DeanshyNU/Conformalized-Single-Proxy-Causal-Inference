#!/usr/bin/env python3
"""
Python version of pred_mgn_est.R with Neural Network replacing Random Forest
Marginally-valid algorithm with estimated bounds
Enables GPU acceleration for faster training and inference
"""

import sys
import os
import numpy as np
import pandas as pd
import torch

# Add parent directory to path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.util_ate import data_gen_ate, conform_score, RegressionForestNN


def main():
    ########################################
    ## Input configurations
    ########################################
    if len(sys.argv) < 6:
        print("Usage: python pred_mgn_est.py <p> <n> <alpha_ind> <Gamma_ind> <seed>")
        sys.exit(1)
    
    p = int(sys.argv[1])
    n = int(sys.argv[2])
    alpha_ind = int(sys.argv[3])
    Gamma_ind = int(sys.argv[4])
    seed = int(sys.argv[5])
    
    alphas = np.arange(0.1, 1.0, 0.1)
    gammas = np.array([1.5, 2, 2.5, 3, 5])
    
    # Coverage target 1-alpha
    alpha = alphas[alpha_ind - 1]  # R uses 1-indexing
    # Confounding level Gamma
    Gamma = gammas[Gamma_ind - 1]  # R uses 1-indexing
    
    data_type = "continuous"
    print(f" - Running the script with marginally-valid algorithm and estimated bounds, "
          f"alpha {alpha}, Gamma {Gamma}, n {n}, p {p}, seed {seed}, type {data_type}")
    
    ########################################
    ## Output directory
    ########################################
    out_dir = "../results/simulation/"
    os.makedirs(out_dir, exist_ok=True)
    
    ########################################
    ## Parameters
    ########################################
    alpha0 = 0
    n_test = 500
    beta_base = np.array([[-0.531], [0.126], [-0.312], [0.018]])
    if p > 4:
        beta = np.vstack([beta_base, np.zeros((p - 4, 1))])
    else:
        beta = beta_base[:p]
    
    noise_level = 0.5
    
    # Generate true probability of treatment
    temp_data = data_gen_ate(n * 1000, p, Gamma, beta, alpha0, obs=False)
    pp = np.mean(temp_data['T'])
    
    # Set seed for reproducibility (same as R version)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    ########################################
    ## Fit on the training fold
    ########################################
    train_data = data_gen_ate(n, p, Gamma, beta, alpha0, obs=True)
    
    # Extract treated samples
    treated_idx = np.where(train_data['T'] == 1)[0][:n]
    train_X = train_data['X'][treated_idx]
    train_Y = train_data['Y1'][treated_idx]
    
    # Train the nonconformity score function (NN instead of quantile_forest)
    print(" - Training quantile regression neural network...")
    device = 'cuda'  # Use GPU
    train_score = conform_score(train_X, train_Y, method='cqr', trained_model=None,
                                quantile=1 - alpha, device=device, epochs=100, 
                                batch_size=32, verbose=False)
    t_mdl = train_score['model']
    
    # Estimate hat{e}(x) using RegressionForestNN (NN instead of regression_forest)
    hat_p = np.mean(train_data['T'])
    print(f" [debug] Training data check:")
    print(f"   hat.p = {hat_p}")
    print(f"   train.data['T'] (first 10): {train_data['T'][:10]}")
    print(f"   train.data['X'] (first row, first 5): {train_data['X'][0, :5]}")
    
    print(" - Training propensity score neural network...")
    e_model = RegressionForestNN(train_data['X'], train_data['T'], device=device,
                                 epochs=100, batch_size=32, verbose=False)
    
    # Check e_model predictions
    train_ex_check = e_model.predict(train_data['X'][:5])['predictions']
    print(f"   e.model predictions (first 5): {train_ex_check}")
    
    ########################################
    ## Calibration
    ########################################
    calib_data = data_gen_ate(n, p, Gamma, beta, alpha0, obs=True)
    
    # Extract treated samples for calibration
    calib_treated_idx = np.where(calib_data['T'] == 1)[0][:n]
    calib_X = calib_data['X'][calib_treated_idx]
    calib_Y = calib_data['Y1'][calib_treated_idx]
    
    # Predict e(x) for calibration set
    calib_ex = e_model.predict(calib_X)['predictions']
    n_calib = len(calib_Y)
    
    # Debug output: check calibration e(x) predictions
    print(f" [debug] Calibration e(x) check:")
    print(f"   calib.ex (first 5): {calib_ex[:5]}")
    print(f"   calib.ex (min, max, mean): {np.min(calib_ex)}, {np.max(calib_ex)}, {np.mean(calib_ex)}")
    
    # Debug output: check e(x) distribution
    print(f" [debug] NN e(x) distribution:")
    print(f"   min: {np.min(calib_ex)}, max: {np.max(calib_ex)}")
    print(f"   mean: {np.mean(calib_ex)}, sd: {np.std(calib_ex)}")
    print(f"   < 0.1: {np.sum(calib_ex < 0.1)}, > 0.9: {np.sum(calib_ex > 0.9)}")
    print(f"   < 0.01: {np.sum(calib_ex < 0.01)}, > 0.99: {np.sum(calib_ex > 0.99)}")
    
    # Lower and upper bounds of weight function
    calib_lx = hat_p * (1 + (1 - calib_ex) / (calib_ex * Gamma))
    calib_ux = hat_p * (1 + Gamma * (1 - calib_ex) / calib_ex)
    calib_nc_w = hat_p / calib_ex
    
    # Debug output: check weights
    print(f" [debug] Gamma = {Gamma}")
    print(f" [debug] First 5 weights comparison:")
    print(f"   calib.lx (head): {calib_lx[:5]}")
    print(f"   calib.ux (head): {calib_ux[:5]}")
    print(f"   calib.wx (head): {calib_nc_w[:5]}")
    print(f"   ratio lx/wx (head): {(calib_lx / calib_nc_w)[:5]}")
    print(f"   ratio lx/wx (mean): {np.mean(calib_lx / calib_nc_w)}, sd: {np.std(calib_lx / calib_nc_w)}")
    
    # Non-conformity score
    calib_score = conform_score(calib_X, calib_Y, method='cqr', trained_model=t_mdl,
                                quantile=1 - alpha, device=device)['score']
    
    # Create DataFrame and sort by score
    calib_all = pd.DataFrame({
        'score': calib_score,
        'lx': calib_lx,
        'ux': calib_ux,
        'wx': calib_nc_w,
        'ex': calib_ex
    })
    calib_all = calib_all.sort_values('score').reset_index(drop=True)
    
    # Calculate estimation error and actual gap
    calib_true_exu = calib_data['exu'][calib_treated_idx]
    calib_true_ex = calib_data['ex'][calib_treated_idx]
    calib_true_wx = pp / calib_true_exu
    
    gap = (np.mean(np.maximum(calib_lx - calib_true_wx, 0)) + 
           np.mean(np.maximum(calib_true_wx - calib_ux, 0)) + 
           1 / len(calib_X) * np.mean(calib_true_wx * np.max(np.maximum(calib_true_wx - calib_ux, 0)))) * np.max(1 / calib_lx)
    
    l_err = np.mean(np.abs(calib_lx - pp * (1 + (1 - calib_true_ex) / (calib_true_ex * Gamma))))
    u_err = np.mean(np.abs(calib_ux - pp * (1 + Gamma * (1 - calib_true_ex) / calib_true_ex)))
    l_inv = np.mean(1 / calib_lx)
    
    ########################################
    ## Generate test fold
    ########################################
    test_data = data_gen_ate(n_test, p, Gamma, beta, alpha0, obs=False)
    test_X = test_data['X']
    test_Y1 = test_data['Y1']
    
    # Predict e(x) for test set
    test_ex = e_model.predict(test_X)['predictions']
    test_lx = hat_p * (1 + (1 - test_ex) / (test_ex * Gamma))
    test_ux = hat_p * (1 + Gamma * (1 - test_ex) / test_ex)
    test_wx = hat_p / test_ex
    
    # Debug output: check test weights
    print(f" [debug] Test weights comparison:")
    print(f"   test.lx (head): {test_lx[:5]}")
    print(f"   test.ux (head): {test_ux[:5]}")
    print(f"   test.wx (head): {test_wx[:5]}")
    print(f"   ratio lx/wx (head): {(test_lx / test_wx)[:5]}")
    print(f"   ratio lx/wx (mean): {np.mean(test_lx / test_wx)}, sd: {np.std(test_lx / test_wx)}")
    
    # Predict quantiles
    test_pred = t_mdl.predict(test_X, quantile=[alpha / 2, 1 - alpha / 2])
    if isinstance(test_pred, dict) and 'predictions' in test_pred:
        test_pred = test_pred['predictions']
    
    ########################################
    ## The confounding-aware algorithm
    ########################################
    print(" - Computing the robust weighted conformal inference...", end="")
    
    # Partial sums for confounding-aware
    sum_num = np.zeros(n_calib)
    sum_den = np.zeros(n_calib)
    sum_num[0] = calib_all['lx'].values[0]
    sum_den[0] = calib_all['lx'].values[0] + np.sum(calib_all['ux'].values[1:])
    
    for k in range(1, n_calib):
        sum_num[k] = sum_num[k - 1] + calib_all['lx'].values[k]
        sum_den[k] = sum_den[k - 1] - calib_all['ux'].values[k] + calib_all['lx'].values[k]
    
    # Confounding-aware prediction
    c_test_lo = np.zeros(n_test)
    c_test_hi = np.zeros(n_test)
    
    for ii in range(n_test):
        ratios = sum_num / (sum_den + test_ux[ii])
        kstar = np.where(ratios > 1 - alpha)[0][0]  # First index where ratio > 1-alpha
        v_kstar = calib_all['score'].values[kstar]
        c_test_lo[ii] = test_pred[ii, 0] - v_kstar
        c_test_hi[ii] = test_pred[ii, 1] + v_kstar
    
    # Evaluate coverage on test data
    c_cover = (c_test_lo <= test_Y1) & (c_test_hi >= test_Y1)
    
    print("Done.")
    
    ########################################
    ## The confounding-unaware algorithm
    ########################################
    print(" - Computing the vanilla weighted conformal inference...", end="")
    
    nc_sum = np.zeros(n_calib)
    nc_sum[0] = calib_all['wx'].values[0]
    
    for k in range(1, n_calib):
        nc_sum[k] = nc_sum[k - 1] + calib_all['wx'].values[k]
    
    nc_test_lo = np.zeros(n_test)
    nc_test_hi = np.zeros(n_test)
    nc_test_weight = hat_p / test_ex
    
    for ii in range(n_test):
        nc_ratios = nc_sum / (nc_sum[n_calib - 1] + nc_test_weight[ii])
        nc_kstar = np.where(nc_ratios > 1 - alpha)[0][0]
        nc_v_kstar = calib_all['score'].values[nc_kstar]
        nc_test_lo[ii] = test_pred[ii, 0] - nc_v_kstar
        nc_test_hi[ii] = test_pred[ii, 1] + nc_v_kstar
    
    nc_cover = (nc_test_lo <= test_Y1) & (nc_test_hi >= test_Y1)
    
    print("Done.")
    
    ########################################
    ## Output summary of test
    ########################################
    res = pd.DataFrame({
        'c.cov': [np.mean(c_cover)],
        'c.len': [np.mean(c_test_hi - c_test_lo)],
        'nc.cov': [np.mean(nc_cover)],
        'nc.len': [np.mean(nc_test_hi - nc_test_lo)],
        'gap': [gap],
        'l_err': [l_err],
        'u_err': [u_err],
        'l_inv': [l_inv],
        'n': [n],
        'p': [p],
        'n_calib': [n_calib],
        'gamma': [Gamma],
        'alpha': [alpha],
        'seed': [seed],
        'data_type': [data_type],
        'method': ['marginal_est']
    })
    
    save_path = f"{out_dir}pred_marginal_est_p_{p}_n_{n}_alpha_{alpha_ind}_gamma_{Gamma_ind}_seed_{seed}_{data_type}_nn.csv"
    res.to_csv(save_path, index=False)
    
    print(f"\nResults saved to: {save_path}")
    print(f"  Confounding-aware coverage: {np.mean(c_cover):.4f}")
    print(f"  Confounding-unaware coverage: {np.mean(nc_cover):.4f}")
    print(f"  Confounding-aware interval length: {np.mean(c_test_hi - c_test_lo):.4f}")
    print(f"  Confounding-unaware interval length: {np.mean(nc_test_hi - nc_test_lo):.4f}")


if __name__ == "__main__":
    main()


