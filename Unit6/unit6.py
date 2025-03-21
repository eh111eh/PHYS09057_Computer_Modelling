import numpy as np
import matplotlib.pyplot as plt
from unit4 import Likelihood
from unit5 import Metropolis

def compute_acceptance_rate(likelihood_function, num_chains=10):
    """
    Run multiple Metropolis chains and compute acceptance rates.
    """
    acceptance_rates = []
    for _ in range(num_chains):
        sampler = Metropolis(
            likelihood_function=likelihood_function,
            initial_params=[0.3, 0.7, 70.0],
            step_sizes=np.array([0.07, 0.1, 0.3]),
            num_samples=6000
        )
        sampler.run()
        acceptance_rate = sampler.acceptance_rate
        acceptance_rates.append(acceptance_rate)
    
    avg_acceptance_rate = np.mean(acceptance_rates)
    print(f"Average Acceptance Rate: {avg_acceptance_rate:.3f}")
    return acceptance_rates

def compute_gelman_rubin_statistic(chains):
    """
    Compute the Gelman-Rubin statistic for each parameter.
    """
    M = len(chains)
    N = len(chains[0])
    
    chains = np.array(chains)
    means = np.mean(chains, axis=1)  # Mean per chain
    variances = np.var(chains, axis=1, ddof=1)  # Variance per chain
    
    B = np.var(means, axis=0, ddof=1)  # Variance of the means
    W = np.mean(variances, axis=0)  # Mean of the variances
    
    R = np.sqrt(((N - 1) / N) + ((M + 1) / M) * (B / W)) # Gelman-Rubin statistic
    return R - 1

burn_in = 1000

def run_multiple_chains_and_compute_r(likelihood_function, num_chains=10, num_samples=6000):
    """
    Run multiple Metropolis chains and compute the Gelman-Rubin statistic progressively.
    """
    chains = []
    r_values_progression = []

    for i in range(1, num_chains + 1):
        sampler = Metropolis(
            likelihood_function=likelihood_function,
            initial_params=[0.3, 0.7, 70.0],
            step_sizes=np.array([0.07, 0.1, 0.3]),
            num_samples=num_samples
        )
        samples = sampler.run()
        chains.append(samples[burn_in:])
        
        if i > 1:  # Compute R only if we have more than 1 chain
            r_values = compute_gelman_rubin_statistic(chains)
            r_values_progression.append(r_values)

    return chains, r_values_progression

def compute_gelman_rubin_statistic_progression(chains):
    """
    Compute the Gelman-Rubin statistic as a function of sample size.
    Returns two separate lists: sample sizes and corresponding R values.
    """
    num_samples = chains.shape[1]
    sample_sizes = np.linspace(500, num_samples, 10, dtype=int)  # 10 checkpoints
    r_values_progression = []

    for size in sample_sizes:
        trimmed_chains = chains[:, :size, :]
        r_values = compute_gelman_rubin_statistic(trimmed_chains)
        r_values_progression.append(r_values)  # Store R values directly

    return sample_sizes, np.array(r_values_progression)

def main():
    data_file = "pantheon_data.txt"
    likelihood = Likelihood(data_file)
    acceptance_rates = compute_acceptance_rate(likelihood, num_chains=10)

    chains, r_values_progression = run_multiple_chains_and_compute_r(likelihood, num_chains=10, num_samples=6000)
    print("Final Gelman-Rubin Statistic per Parameter:", r_values_progression[-1])

    # Plot Chain Convergence
    param_names = ["Omega_m", "Omega_lambda", "H0"]
    chains = np.array(chains)

    fig, axes = plt.subplots(len(param_names), 1, figsize=(10, 7), sharex=True)
    for i, ax in enumerate(axes):
        for chain in chains:
            ax.plot(chain[:, i], alpha=0.6, linewidth=0.8)
        ax.set_ylabel(param_names[i])
        ax.grid(True)
    axes[-1].set_xlabel("Sample Index")
    fig.suptitle("Convergence of Chains for Each Parameter", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Plot R vs Number of Chains
    fig, ax = plt.subplots(figsize=(8, 5))
    x_vals = np.arange(2, len(r_values_progression) + 2)  # X-axis: number of chains (starting from 2)
    
    for i, param in enumerate(param_names):
        r_vals = [r[i] for r in r_values_progression]  # Extract R values for this parameter
        ax.plot(x_vals, r_vals, marker='o', linestyle='-', label=param)

    ax.axhline(y=0.1, color='r', linestyle='--', label="Threshold (0.1)")
    ax.set_xlabel("Number of Chains")
    ax.set_ylabel("Gelman-Rubin Statistic (R - 1)")
    ax.set_title("Convergence of Gelman-Rubin Statistic")
    ax.legend()
    ax.grid(True)
    plt.show()

    # Plot R vs Number of Samples
    sample_sizes, r_progression = compute_gelman_rubin_statistic_progression(chains)
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, param in enumerate(param_names):
        ax.plot(sample_sizes, r_progression[:, i], marker='o', linestyle='-', label=param)

    ax.axhline(y=0.1, color='r', linestyle='--', label="Threshold (0.1)")
    ax.set_xlabel("Number of Samples per Chain")
    ax.set_ylabel("Gelman-Rubin Statistic (R - 1)")
    ax.set_title("Convergence of Gelman-Rubin Statistic vs. Number of Samples")
    ax.legend()
    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
