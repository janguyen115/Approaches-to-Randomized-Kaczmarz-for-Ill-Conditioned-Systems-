"""
Coreset Extraction for Overdetermined Linear Systems
----------------------------------------------------
This script reproduces two figures:
1. Relative solution error vs. coreset factor c.
2. Residual norms (full vs. coreset) vs. c (log scale).

Author: Oleg Presnyakov
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_ill_conditioned_matrix(m, n, cond_num=1e7, seed=None):
    """Generate an m×n ill-conditioned matrix using truncated SVD."""
    rng = np.random.default_rng(seed)
    # Create orthogonal factors (economy QR for speed)
    Q1, _ = np.linalg.qr(rng.standard_normal((m, n)))
    Q2, _ = np.linalg.qr(rng.standard_normal((n, n)))
    # Singular values: geometric progression from cond_num to 1
    s = np.geomspace(cond_num, 1, num=n)
    S = np.diag(s)
    A = Q1 @ S @ Q2.T
    return A


def coreset_least_squares(m=800, n=50, c=2, cond_num=1e7, seed=None):
    """Run the coreset least-squares experiment."""
    rng = np.random.default_rng(seed)
    A = generate_ill_conditioned_matrix(m, n, cond_num, seed)
    x_true = rng.standard_normal(n)
    b = A @ x_true + 0.01 * rng.standard_normal(m)  # add small noise

    # Step 1: Row scoring
    scores = np.abs(A @ x_true)
    # Step 2: Coreset extraction
    k = int(c * n)
    idx = np.argsort(scores)[:k]
    B = A[idx, :]
    b_B = b[idx]

    # Step 3: Least-squares solves
    x_A, *_ = np.linalg.lstsq(A, b, rcond=None)
    x_B, *_ = np.linalg.lstsq(B, b_B, rcond=None)

    # Step 4: Evaluation metrics
    rel_error = np.linalg.norm(x_B - x_A) / np.linalg.norm(x_A)
    residual_full = np.linalg.norm(A @ x_B - b)
    residual_core = np.linalg.norm(B @ x_B - b_B)

    return rel_error, residual_full, residual_core


def run_experiment():
    """Generate plots for different c and condition numbers."""
    c_values = [1, 2, 3, 4, 5]
    cond_values = [1e3, 1e5, 1e7]
    results = {}

    for cond in cond_values:
        rel_errors, full_res, core_res = [], [], []
        for c in c_values:
            re, rf, rc = coreset_least_squares(m=800, n=50, c=c, cond_num=cond, seed=42)
            rel_errors.append(re)
            full_res.append(rf)
            core_res.append(rc)
        results[cond] = (rel_errors, full_res, core_res)

    # --- Plot 1: Relative error vs c ---
    plt.figure(figsize=(7, 5))
    for cond in cond_values:
        plt.plot(c_values, results[cond][0], marker='o', label=f'cond={cond:.0e}')
    plt.xlabel('Coreset factor c', fontsize=12)
    plt.ylabel(r'Relative error $\|x_B - x_A\| / \|x_A\|$', fontsize=12)
    plt.title('Relative Solution Error vs. Coreset Size', fontsize=13)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("relative_error_vs_c.png", dpi=300)
    plt.show()

    # --- Plot 2: Residual norms (log scale) ---
    plt.figure(figsize=(7, 5))
    for cond in cond_values:
        plt.plot(c_values, results[cond][1], '--', marker='x', label=f'Full residual, cond={cond:.0e}')
        plt.plot(c_values, results[cond][2], '-', marker='o', label=f'Coreset residual, cond={cond:.0e}')
    plt.xlabel('Coreset factor c', fontsize=12)
    plt.ylabel('Residual norm', fontsize=12)
    plt.title('Residual Norms: Full vs. Coreset System', fontsize=13)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig("residual_norms_vs_c.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    run_experiment()
