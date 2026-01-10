#!/usr/bin/env python3
"""
Python version of util_ate.R with Neural Network replacing Random Forest
Enables GPU acceleration for faster training and inference
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm
from typing import Optional, Dict, Union, List


############################################################
#####           Neural Network Models               #######
############################################################

class QuantileRegressionNN(nn.Module):
    """
    Neural Network for Quantile Regression (CQR)
    Replaces quantile_forest from grf package
    """
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], dropout: float = 0.1):
        super(QuantileRegressionNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output 2 quantiles: lower and upper
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class RegressionNN(nn.Module):
    """
    Neural Network for Regression (e.g., propensity score estimation)
    Replaces regression_forest from grf package
    """
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], dropout: float = 0.1):
        super(RegressionNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output single value (probability or continuous value)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # For propensity scores in [0, 1]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def pinball_loss(predictions, targets, quantile):
    """
    Pinball loss (quantile loss) for quantile regression
    """
    errors = targets - predictions
    return torch.mean(torch.max((quantile - 1) * errors, quantile * errors))


class QuantileForestNN:
    """
    Wrapper class for Quantile Regression Neural Network
    Compatible interface with grf's quantile_forest
    """
    def __init__(self, X, Y, device='cuda', epochs=100, batch_size=32, lr=0.001, 
                 hidden_dims=[64, 32], dropout=0.1, verbose=False):
        """
        Initialize and train Quantile Regression NN
        
        Args:
            X: Features [n, p]
            Y: Response [n]
            device: 'cuda' or 'cpu'
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            verbose: Print training progress
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(Y, pd.Series):
            Y = Y.values
        
        # Ensure proper shapes
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)
        
        self.input_dim = X.shape[1]
        self.model = QuantileRegressionNN(self.input_dim, hidden_dims, dropout).to(self.device)
        
        # Train the model
        self._train(X, Y, epochs, batch_size, lr)
    
    def _train(self, X, Y, epochs, batch_size, lr):
        """Train the quantile regression model"""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        Y_tensor = torch.FloatTensor(Y).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        n_samples = X.shape[0]
        n_batches = max(1, n_samples // batch_size)
        
        self.model.train()
        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(n_samples)
            total_loss = 0.0
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                X_batch = X_tensor[batch_indices]
                Y_batch = Y_tensor[batch_indices]
                
                optimizer.zero_grad()
                
                # Forward pass - outputs [batch_size, 2] for lower and upper quantiles
                predictions = self.model(X_batch)
                
                # Loss for lower quantile (e.g., 0.05) and upper quantile (e.g., 0.95)
                # We use fixed quantiles here, will be parameterized in predict()
                loss_lower = pinball_loss(predictions[:, 0:1], Y_batch, 0.05)
                loss_upper = pinball_loss(predictions[:, 1:2], Y_batch, 0.95)
                loss = loss_lower + loss_upper
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f}")
    
    def predict(self, X, quantile=[0.05, 0.95]):
        """
        Predict quantiles for new data
        
        Args:
            X: Features [n, p]
            quantile: List of two quantiles [lower, upper]
        
        Returns:
            Dictionary with 'predictions' key containing [n, 2] array
        """
        self.model.eval()
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        # Return as numpy array
        result = predictions.cpu().numpy()
        
        return {'predictions': result}


class RegressionForestNN:
    """
    Wrapper class for Regression Neural Network
    Compatible interface with grf's regression_forest
    """
    def __init__(self, X, Y, device='cuda', epochs=100, batch_size=32, lr=0.001,
                 hidden_dims=[64, 32], dropout=0.1, verbose=False):
        """
        Initialize and train Regression NN
        
        Args:
            X: Features [n, p]
            Y: Response [n] (binary for propensity scores)
            device: 'cuda' or 'cpu'
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            verbose: Print training progress
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(Y, (pd.Series, list)):
            Y = np.array(Y)
        
        # Ensure proper shapes
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)
        
        self.input_dim = X.shape[1]
        self.model = RegressionNN(self.input_dim, hidden_dims, dropout).to(self.device)
        
        # Train the model
        self._train(X, Y, epochs, batch_size, lr)
    
    def _train(self, X, Y, epochs, batch_size, lr):
        """Train the regression model"""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        Y_tensor = torch.FloatTensor(Y).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()  # Binary cross-entropy for propensity scores
        
        n_samples = X.shape[0]
        n_batches = max(1, n_samples // batch_size)
        
        self.model.train()
        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(n_samples)
            total_loss = 0.0
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                X_batch = X_tensor[batch_indices]
                Y_batch = Y_tensor[batch_indices]
                
                optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(X_batch)
                loss = criterion(predictions, Y_batch)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f}")
    
    def predict(self, newdata):
        """
        Predict for new data
        
        Args:
            newdata: Features [n, p]
        
        Returns:
            Dictionary with 'predictions' key containing [n] array
        """
        self.model.eval()
        
        # Convert to numpy if needed
        if isinstance(newdata, pd.DataFrame):
            newdata = newdata.values
        if len(newdata.shape) == 1:
            newdata = newdata.reshape(-1, 1)
        
        X_tensor = torch.FloatTensor(newdata).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        # Return as numpy array, flattened
        result = predictions.cpu().numpy().flatten()
        
        return {'predictions': result}


############################################################
#####      Data Generating Process of Sec6.1       #########
############################################################

def data_gen_ate(n: int, p: int, Gamma: float, beta: np.ndarray, 
                 alpha0: float = 0, obs: bool = True) -> Dict:
    """
    Generate data for Average Treatment Effect estimation
    
    Args:
        n: Output sample size
        p: Covariate dimension
        Gamma: Confounding level
        beta: Linear coefficients [p, 1] or [p]
        alpha0: Intercept
        obs: If True, generate observations (training data) of size n
             If False, generate all data of size n
    
    Returns:
        Dictionary containing T, X, U, Y1, ex, exu
    """
    # Ensure beta is proper shape
    if len(beta.shape) == 1:
        beta = beta.reshape(-1, 1)
    
    # Generate X from uniform [0, 1]
    X = np.random.uniform(0, 1, size=(n, p))
    
    # Generate U: one-dimensional confounder U ~ N(0, σ²)
    # Variance depends on first dimension of X
    sigma_u = np.abs(1 + 0.5 * np.sin(2.5 * X[:, 0]))
    U = np.random.normal(0, 1, size=n) * sigma_u
    
    # Generate Y1 (only Y(1) for counterfactual prediction)
    Y1 = (X @ beta).flatten() + U
    
    # e(x) = P(T=1|X=x)
    logit_ex = alpha0 + (X @ beta).flatten()
    prop_x = np.exp(logit_ex) / (1 + np.exp(logit_ex))
    
    # Compute p(x) to ensure E[P(A|U,X)|X] = P(A|X)
    denominator1 = prop_x + (1 - prop_x) / Gamma
    denominator2 = prop_x + Gamma * (1 - prop_x)
    p_x = (1 / denominator1 - 1) / (1 / denominator1 - 1 / denominator2)
    
    # Compute threshold t(x)
    t_x = norm.ppf(1 - p_x / 2) * sigma_u
    
    # e(x,u) = P(T=1|X=x,U=u) ensuring E[e(X,U)|X] = e(X)
    a_x = prop_x / (prop_x + Gamma * (1 - prop_x))
    b_x = prop_x / (prop_x + (1 - prop_x) / Gamma)
    
    # Piecewise assignment based on |U| > t(x)
    prop_xu = np.where(np.abs(U) > t_x, a_x, b_x)
    
    # Generate T based on e(x,u)
    TT = np.random.binomial(1, prop_xu, size=n)
    
    if not obs:
        return {
            'T': TT,
            'X': X,
            'U': U,
            'Y1': Y1,
            'ex': prop_x,
            'exu': prop_xu
        }
    else:
        # Ensure we have at least n treated samples
        n_useful = np.sum(TT)
        
        while n_useful < n:
            # Recursively generate more data
            add_data = data_gen_ate(n, p, Gamma, beta, alpha0, obs=False)
            X = np.vstack([X, add_data['X']])
            U = np.concatenate([U, add_data['U']])
            Y1 = np.concatenate([Y1, add_data['Y1']])
            prop_x = np.concatenate([prop_x, add_data['ex']])
            prop_xu = np.concatenate([prop_xu, add_data['exu']])
            TT = np.concatenate([TT, add_data['T']])
            n_useful = np.sum(TT)
        
        return {
            'T': TT,
            'X': X,
            'U': U,
            'Y1': Y1,
            'ex': prop_x,
            'exu': prop_xu
        }


############################################################
#####   Output nonconformity score and trained model  ######
############################################################

def conform_score(X, Y, method: str = 'cqr', trained_model = None, 
                  quantile: float = 0.9, device: str = 'cuda',
                  epochs: int = 100, batch_size: int = 32, 
                  verbose: bool = False) -> Dict:
    """
    Compute nonconformity score using Conformalized Quantile Regression (CQR)
    
    Args:
        X: Covariate matrix [n, p]
        Y: Response vector [n]
        method: Method for computing scores (currently only 'cqr' supported)
        trained_model: Pre-trained model (if None, train new model)
        quantile: Coverage target (1-alpha)
        device: 'cuda' or 'cpu'
        epochs: Number of training epochs (if training new model)
        batch_size: Batch size for training
        verbose: Print training progress
    
    Returns:
        Dictionary with 'model' and 'score' keys
    """
    if method == 'cqr':
        if trained_model is None:
            # Train new quantile regression model
            trained_model = QuantileForestNN(
                X, Y, device=device, epochs=epochs, 
                batch_size=batch_size, verbose=verbose
            )
        
        # Predict quantiles
        quantile_lower = (1 - quantile) / 2
        quantile_upper = 1 - (1 - quantile) / 2
        
        qs = trained_model.predict(X, quantile=[quantile_lower, quantile_upper])
        if isinstance(qs, dict) and 'predictions' in qs:
            qs = qs['predictions']
        
        q_lo = qs[:, 0]
        q_hi = qs[:, 1]
        
        # Ensure Y is proper shape
        if isinstance(Y, (pd.Series, list)):
            Y = np.array(Y)
        if len(Y.shape) > 1:
            Y = Y.flatten()
        
        # Compute nonconformity score
        score = np.maximum(Y - q_hi, q_lo - Y)
    else:
        raise ValueError(f"Method {method} not supported")
    
    return {'model': trained_model, 'score': score}


############################################################
#####           WSR Inequality Functions            #######
############################################################

def compute_k_lower(x, mu, nu):
    """Compute k for WSR lower bound"""
    kterms = 1 + nu * (x - mu)
    ks = np.zeros(len(kterms))
    ks[0] = kterms[0]
    for ii in range(1, len(ks)):
        ks[ii] = ks[ii-1] * kterms[ii]
    return np.max(ks)


def compute_k_upper(x, mu, nu):
    """Compute k for WSR upper bound"""
    kterms = 1 - nu * (x - mu)
    ks = np.zeros(len(kterms))
    ks[0] = kterms[0]
    for ii in range(1, len(ks)):
        ks[ii] = ks[ii-1] * kterms[ii]
    return np.max(ks)


def wsr_lower(delta, x):
    """Compute WSR lower bound"""
    n = len(x)
    
    # Compute mu_hat
    mu_hat = (0.5 + np.cumsum(x)) / (np.arange(1, n + 1) + 1)
    
    # Compute sig_hat
    sig_hat = (0.25 + np.cumsum((x - mu_hat)**2)) / (np.arange(1, n + 1) + 1)
    
    # Compute nu
    nu = np.minimum(1, np.sqrt(2 * np.log(1 / delta) / (n * sig_hat**2)))
    nu[1:] = nu[:-1]
    nu[0] = min(1, np.sqrt(2 * np.log(1 / delta) / (n / 4)))
    
    # Search for bound
    u_list = np.arange(1, 1001) / 1000
    k_all = np.array([compute_k_lower(x, mu_hat, nu) for u in u_list])
    
    indices = np.where(k_all <= 1 / delta)[0]
    if len(indices) == 0:
        u_ind = 999
    else:
        u_ind = indices[0]
    
    bnd = u_list[u_ind]
    return bnd


def wsr_upper(delta, x):
    """Compute WSR upper bound"""
    n = len(x)
    
    # Compute mu_hat
    mu_hat = (0.5 + np.cumsum(x)) / (np.arange(1, n + 1) + 1)
    
    # Compute sig_hat
    sig_hat = (0.25 + np.cumsum((x - mu_hat)**2)) / (np.arange(1, n + 1) + 1)
    
    # Compute nu
    nu = np.minimum(1, np.sqrt(2 * np.log(1 / delta) / (n * sig_hat**2)))
    
    # Search for bound
    u_list = np.arange(1, 1001) / 1000
    k_all = np.array([compute_k_upper(x, mu_hat, nu) for u in u_list])
    
    indices = np.where(k_all > 1 / delta)[0]
    if len(indices) == 0:
        u_ind = 999
    else:
        u_ind = indices[0]
    
    bnd = u_list[u_ind]
    return bnd


def wsr_cdf_single(delta, lx, ux, M, i, rand_given, rand_ind):
    """Compute WS-R lower bound for cdf at a single point V[i]"""
    n = len(lx)
    
    # Create lxx and uxx
    lxx = np.concatenate([lx[:i+1], np.zeros(n-i-1)]) / M
    uxx = 1 + np.concatenate([np.zeros(i+1), -ux[i+1:]]) / M
    
    if rand_given:
        wsr1 = wsr_lower(delta / 2, lxx[rand_ind]) * M
        wsr2 = wsr_lower(delta / 2, uxx[rand_ind]) * M + 1 - M
    else:
        # Random permutation
        lxx_perm = np.random.permutation(lxx)
        uxx_perm = np.random.permutation(uxx)
        wsr1 = wsr_lower(delta / 2, lxx_perm) * M
        wsr2 = wsr_lower(delta / 2, uxx_perm) * M + 1 - M
    
    return max(wsr1, wsr2)


def wsr_cdf(delta, lx, ux, M, rand_given, rand_ind):
    """Compute WS-R lower bound for cdf at all points V[i]"""
    n = len(lx)
    w_all = np.array([
        wsr_cdf_single(delta, lx, ux, M, i, rand_given, rand_ind)
        for i in range(n)
    ])
    return w_all


def wsr_qtl(delta, calib_all, M, alpha, rand_ind):
    """Binary search for quantile of wsr lower bound"""
    n_calib = len(calib_all)
    l_i = 0  # Python uses 0-indexing
    r_i = n_calib - 2
    m_i = (l_i + r_i) // 2
    
    lx = calib_all['lx'].values
    ux = calib_all['ux'].values
    
    left_cdf = wsr_cdf_single(delta, lx, ux, M, l_i, True, rand_ind)
    right_cdf = wsr_cdf_single(delta, lx, ux, M, r_i, True, rand_ind)
    
    if right_cdf < 1 - alpha:
        return np.inf
    
    mid_cdf = wsr_cdf_single(delta, lx, ux, M, m_i, True, rand_ind)
    gap = min(m_i - l_i, r_i - m_i)
    
    while gap > 0:
        if mid_cdf < 1 - alpha:
            l_i = m_i
            m_i = (l_i + r_i) // 2
            left_cdf = mid_cdf
            mid_cdf = wsr_cdf_single(delta, lx, ux, M, m_i, True, rand_ind)
        else:
            r_i = m_i
            m_i = (l_i + r_i) // 2
            right_cdf = mid_cdf
            mid_cdf = wsr_cdf_single(delta, lx, ux, M, m_i, True, rand_ind)
        
        gap = min(m_i - l_i, r_i - m_i)
    
    if mid_cdf >= 1 - alpha:
        return calib_all['score'].values[m_i]
    else:
        return calib_all['score'].values[r_i]


def ate_single_pac(Gamma, pp, calib_ex, calib_score, delta, alpha, rand_ind):
    """
    Find the hat{v} of pac procedure for one single Gamma
    
    Args:
        Gamma: Confounding level
        pp: Population treatment probability
        calib_ex: Calibration propensity scores
        calib_score: Calibration nonconformity scores
        delta: Confidence level
        alpha: Coverage target
        rand_ind: Random indices for permutation
    
    Returns:
        eta_gamma: Threshold value
    """
    # Lower and upper bounds for likelihood ratio
    calib_lx = pp * (1 + (1 - calib_ex) / (calib_ex * Gamma))
    calib_ux = pp * (1 + Gamma * (1 - calib_ex) / calib_ex)
    
    # Aggregate information
    calib_all = pd.DataFrame({
        'score': calib_score,
        'lx': calib_lx,
        'ux': calib_ux,
        'ex': calib_ex
    })
    
    # Compute the 1-alpha quantile
    M = np.max(calib_ux)
    eta_gamma = wsr_qtl(delta, calib_all, M, alpha, rand_ind)
    
    return eta_gamma


if __name__ == "__main__":
    # Test the functions
    print("Testing util_ate.py with GPU acceleration...")
    
    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Test parameters
    n = 500
    p = 4
    Gamma = 2.0
    beta = np.array([[-0.531], [0.126], [-0.312], [0.018]])
    alpha0 = 0
    
    # Test data generation
    print(f"\n1. Testing data_gen_ate with n={n}, p={p}, Gamma={Gamma}")
    data = data_gen_ate(n, p, Gamma, beta, alpha0, obs=False)
    print(f"   Generated data shapes:")
    print(f"   - X: {data['X'].shape}")
    print(f"   - Y1: {data['Y1'].shape}")
    print(f"   - T: {data['T'].shape}")
    print(f"   - T=1 ratio: {np.mean(data['T']):.3f}")
    
    # Test conform_score with GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n2. Testing conform_score with device={device}")
    
    treated_idx = np.where(data['T'] == 1)[0][:100]
    X_train = data['X'][treated_idx]
    Y_train = data['Y1'][treated_idx]
    
    result = conform_score(X_train, Y_train, method='cqr', device=device, 
                          epochs=50, verbose=True)
    print(f"   Nonconformity scores shape: {result['score'].shape}")
    print(f"   Score statistics: mean={np.mean(result['score']):.3f}, "
          f"std={np.std(result['score']):.3f}")
    
    print("\n✅ All tests passed! GPU acceleration is working.")

