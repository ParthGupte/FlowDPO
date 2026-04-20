import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def hutchinson_trace(M, k):
    """
    Hutchinson trace estimator
    """
    n = M.shape[0]
    estimate = 0.0

    for _ in range(k):
        eps = np.random.randn(n, 1)
        estimate += eps.T @ M @ eps

    return (estimate / k).item()


def relative_error(est, true):
    return abs(est - true) / abs(true)


###########################################
# Experiment 1: error vs number of samples
###########################################

n = 500
M = np.random.randn(n, n)

true_trace = np.trace(M)

k_values = np.logspace(0, 3, 20).astype(int)
errors_k = []

for k in tqdm(k_values):
    est = hutchinson_trace(M, k)
    errors_k.append(relative_error(est, true_trace))

plt.figure()
plt.plot(k_values, errors_k, marker="o")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Number of Hutchinson samples (k)")
plt.ylabel("Relative error")
plt.title("Hutchinson Trace Estimator Accuracy vs k")
plt.grid(True)
plt.savefig("samples")


###########################################
# Experiment 2: error vs matrix dimension
###########################################

k = 200
n_values = [50, 100, 200, 500, 1000, 2000]
errors_n = []

for n in tqdm(n_values):
    M = np.random.randn(n, n)
    true_trace = np.trace(M)

    est = hutchinson_trace(M, k)
    errors_n.append(relative_error(est, true_trace))

plt.figure()
plt.plot(n_values, errors_n, marker="o")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Matrix size (n)")
plt.ylabel("Relative error")
plt.title(f"Hutchinson Trace Estimator Accuracy vs n (k={k})")
plt.grid(True)
plt.savefig("sizes")