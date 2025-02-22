import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import simpson
from scipy.stats import gaussian_kde
from unit4 import Likelihood

# ------------------ Task 4.1 (Using imshow function) ------------------

def compute_likelihood_grid(data_file, grid_size=10):
    omega_m_vals = np.linspace(0.01, 0.8, grid_size)
    omega_lambda_vals = np.linspace(0.1, 0.8, grid_size)
    h0_vals = np.linspace(69, 72, grid_size)
    
    likelihood = Likelihood(data_file)
    likelihood_grid = np.zeros((grid_size, grid_size, grid_size))

    start_time = time.time()
    
    for i, omega_m in enumerate(omega_m_vals):
        for j, omega_lambda in enumerate(omega_lambda_vals):
            for k, h0 in enumerate(h0_vals):
                likelihood_grid[i, j, k] = np.exp(likelihood([omega_m, omega_lambda, h0]))
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Likelihood calculations took {elapsed_time:.2f} seconds.")
    
    likelihood_grid /= np.sum(likelihood_grid)  # Normalisation
    
    return omega_m_vals, omega_lambda_vals, h0_vals, likelihood_grid

def plot_3d_likelihood(omega_m_vals, omega_lambda_vals, h0_vals, likelihood_grid):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x, y, z = np.meshgrid(omega_m_vals, omega_lambda_vals, h0_vals, indexing='ij')
    sc = ax.scatter(x, y, z, c=likelihood_grid.ravel(), cmap='viridis', alpha=0.7)
    
    ax.set_xlabel("Omega m")
    ax.set_ylabel("Omega lambda")
    ax.set_zlabel("H0")
    ax.set_title("3D Normalised Likelihood Grid")
    fig.colorbar(sc, label="Normalised Likelihood")
    plt.show()

def marginalize_over_omega_m(likelihood_grid):
    return simpson(likelihood_grid, axis=0)

def marginalize_over_omega_lambda(likelihood_grid):
    return simpson(likelihood_grid, axis=1)

def marginalize_over_h0(likelihood_grid):
    return simpson(likelihood_grid, axis=2)

def marginalize_to_1d(likelihood_grid, axis1, axis2):
    marginalized = simpson(simpson(likelihood_grid, axis=axis1), axis=axis2 - 1)
    return marginalized / np.max(marginalized)

def plot_1d_marginalized_likelihood(omega_m_vals, omega_lambda_vals, h0_vals, likelihood_grid):
    marginalized_omega_m = marginalize_to_1d(likelihood_grid, axis1=1, axis2=2)
    marginalized_omega_lambda = marginalize_to_1d(likelihood_grid, axis1=0, axis2=2)
    marginalized_h0 = marginalize_to_1d(likelihood_grid, axis1=0, axis2=1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    zoom_range_m = (0.01, 0.8)
    zoom_range_lambda = (0.1, 0.8)
    zoom_range_h0 = (69, 72)
    
    # Plot Omega_m marginalized likelihood
    axes[0].plot(omega_m_vals, marginalized_omega_m, label=r"$\Omega_m$", color='b')
    axes[0].set_xlabel(r"$\Omega_m$")
    axes[0].set_ylabel("Normalised Likelihood")
    axes[0].set_title(r"Marginalised over $\Omega_m$")
    axes[0].set_xlim(zoom_range_m)
    axes[0].grid()
    
    # Plot Omega_lambda marginalized likelihood
    axes[1].plot(omega_lambda_vals, marginalized_omega_lambda, label=r"$\Omega_\Lambda$", color='g')
    axes[1].set_xlabel(r"$\Omega_\Lambda$")
    axes[1].set_ylabel("Normalised Likelihood")
    axes[1].set_title(r"Marginalised over $\Omega_\Lambda$")
    axes[1].set_xlim(zoom_range_lambda)
    axes[1].grid()
    
    # Plot H0 marginalized likelihood
    axes[2].plot(h0_vals, marginalized_h0, label=r"$H_0$", color='r')
    axes[2].set_xlabel(r"$H_0$")
    axes[2].set_ylabel("Normalised Likelihood")
    axes[2].set_title(r"Marginalised over $H_0$")
    axes[2].set_xlim(zoom_range_h0)
    axes[2].grid()
    
    plt.tight_layout()
    plt.show()

def plot_2d_marginalized_likelihood(omega_m_vals, omega_lambda_vals, h0_vals, likelihood_grid):
    marginalized_omega_m = marginalize_over_omega_m(likelihood_grid)
    marginalized_omega_lambda = marginalize_over_omega_lambda(likelihood_grid)
    marginalized_h0 = marginalize_over_h0(likelihood_grid)
    
    marginalized_omega_m /= np.max(marginalized_omega_m)
    marginalized_omega_lambda /= np.max(marginalized_omega_lambda)
    marginalized_h0 /= np.max(marginalized_h0)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cmap = 'viridis'

    """
    # -------- Using pyplot (pcolormesh) --------
    
    X, Y = np.meshgrid(omega_lambda_vals, h0_vals)
    pcm0 = axes[0].pcolormesh(X, Y, marginalized_omega_m.T, cmap=cmap, shading='auto')
    axes[0].set_xlabel("Omega lambda")
    axes[0].set_ylabel("H0")
    axes[0].set_title("Marginalised over Omega m")
    fig.colorbar(pcm0, ax=axes[0], label="Normalised Likelihood")
    
    X, Y = np.meshgrid(omega_m_vals, h0_vals)
    pcm1 = axes[1].pcolormesh(X, Y, marginalized_omega_lambda.T, cmap=cmap, shading='auto')
    axes[1].set_xlabel("Omega m")
    axes[1].set_ylabel("H0")
    axes[1].set_title("Marginalised over Omega lambda")
    fig.colorbar(pcm1, ax=axes[1], label="Normalised Likelihood")
    
    X, Y = np.meshgrid(omega_m_vals, omega_lambda_vals)
    pcm2 = axes[2].pcolormesh(X, Y, marginalized_h0.T, cmap=cmap, shading='auto')
    axes[2].set_xlabel("Omega m")
    axes[2].set_ylabel("Omega lambda")
    axes[2].set_title("Marginalised over H0")
    fig.colorbar(pcm2, ax=axes[2], label="Normalised Likelihood")
    
    plt.tight_layout()
    plt.show()
    """
    
    im0 = axes[0].imshow(marginalized_omega_m, extent=[omega_lambda_vals[0], omega_lambda_vals[-1], h0_vals[0], h0_vals[-1]],
                   origin='lower', aspect='auto', cmap=cmap)
    axes[0].set_xlabel("Omega lambda")
    axes[0].set_ylabel("H0")
    axes[0].set_title("Marginalised over Omega m")
    fig.colorbar(im0, ax=axes[0], label="Normalised Likelihood")
    
    im1 = axes[1].imshow(marginalized_omega_lambda, extent=[omega_m_vals[0], omega_m_vals[-1], h0_vals[0], h0_vals[-1]],
                   origin='lower', aspect='auto', cmap=cmap)
    axes[1].set_xlabel("Omega m")
    axes[1].set_ylabel("H0")
    axes[1].set_title("Marginalised over Omega lambda")
    fig.colorbar(im1, ax=axes[1], label="Normalised Likelihood")
    
    im2 = axes[2].imshow(marginalized_h0, extent=[omega_m_vals[0], omega_m_vals[-1], omega_lambda_vals[0], omega_lambda_vals[-1]],
                   origin='lower', aspect='auto', cmap=cmap)
    axes[2].set_xlabel("Omega m")
    axes[2].set_ylabel("Omega lambda")
    axes[2].set_title("Marginalised over H0")
    fig.colorbar(im2, ax=axes[2], label="Normalised Likelihood")
    
    plt.tight_layout()
    plt.show()

# ------------------ Task 4.2 Metropolis ------------------
class Metropolis:
    def __init__(self, likelihood_function, initial_params, step_sizes, num_samples, burn_in=1000):
        self.likelihood_function = likelihood_function
        self.current_params = np.array(initial_params)
        self.step_sizes = np.array(step_sizes)
        self.num_samples = num_samples
        self.burn_in = burn_in
        self.samples = []

    def propose_new_params(self):
        new_params = self.current_params + self.step_sizes * np.random.randn(len(self.current_params))
        
        new_params[0] = max(min(new_params[0], 0.6), 0.01)   # Omega_m in [0.01, 0.6]
        new_params[1] = max(min(new_params[1], 0.8), 0.2)    # Omega_lambda in [0.2, 0.8]
        new_params[2] = max(min(new_params[2], 72), 69)      # H0 in [69, 72]
        
        return new_params

    def run(self):
        accepted = 0
        log_likelihood_current = self.likelihood_function(self.current_params)

        for i in range(self.num_samples + self.burn_in):
            new_params = self.propose_new_params()

            # Compute log-likelihoods
            log_likelihood_new = self.likelihood_function(new_params)

            # Acceptance criteria
            log_alpha = log_likelihood_new - log_likelihood_current
            if np.log(np.random.rand()) < log_alpha:
                self.current_params = new_params
                log_likelihood_current = log_likelihood_new
                accepted += 1
            
            if i >= self.burn_in:
                self.samples.append(self.current_params.copy())

        print(f"Acceptance Rate: {accepted / (self.num_samples + self.burn_in):.3f}")
        return np.array(self.samples)

def plot_1d_histograms(samples):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    params = ["Omega_m", "Omega_lambda", "H0"]
    
    for i, param in enumerate(params):
        axes[i].hist(samples[:, i], bins=30, density=True, alpha=0.7, color="blue")
        axes[i].set_xlabel(param)
        axes[i].set_ylabel("Density")
        axes[i].set_title(f"Histogram of {param}")

    plt.tight_layout()
    plt.show()

def plot_2d_scatter(samples):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    pairs = [(0, 1, "Omega_m", "Omega_lambda"),
             (0, 2, "Omega_m", "H0"),
             (1, 2, "Omega_lambda", "H0")]

    for i, (x_idx, y_idx, x_label, y_label) in enumerate(pairs):
        x = samples[:, x_idx]
        y = samples[:, y_idx]

        # Compute the density estimate
        xy = np.vstack([x, y])
        density = gaussian_kde(xy)(xy)  # Kernel Density Estimation

        sc = axes[i].scatter(x, y, c=density, cmap='viridis', alpha=0.5)
        axes[i].set_xlabel(x_label)
        axes[i].set_ylabel(y_label)
        axes[i].set_title(f"{x_label} vs {y_label}")

        cbar = fig.colorbar(sc, ax=axes[i])
        cbar.set_label("Probability Density")

    plt.tight_layout()
    plt.show()

def plot_3d_chain(samples):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = samples[:, 0], samples[:, 1], samples[:, 2]

    xyz = np.vstack([x, y, z])
    density = gaussian_kde(xyz)(xyz)

    sc = ax.scatter(x, y, z, c=density, cmap='viridis', alpha=0.5)

    ax.set_xlabel("Omega_m")
    ax.set_ylabel("Omega_lambda")
    ax.set_zlabel("H0")
    ax.set_title("Metropolis MCMC Chain")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Probability Density")

    plt.show()

def main():
    data_file = "pantheon_data.txt"

    # Compute likelihood grid (Task 4.1)
    grid_size = 30
    omega_m_vals, omega_lambda_vals, h0_vals, likelihood_grid = compute_likelihood_grid(data_file, grid_size)
    
    plot_3d_likelihood(omega_m_vals, omega_lambda_vals, h0_vals, likelihood_grid)
    plot_2d_marginalized_likelihood(omega_m_vals, omega_lambda_vals, h0_vals, likelihood_grid)
    plot_1d_marginalized_likelihood(omega_m_vals, omega_lambda_vals, h0_vals, likelihood_grid)

    # Perform Metropolis (Task 4.2)
    likelihood = Likelihood(data_file)
    sampler = Metropolis(
        likelihood_function=likelihood,
        initial_params=[0.3, 0.7, 70.0],
        step_sizes = np.array([0.01, 0.02, 0.1]),
        num_samples=10000
    )
    samples = sampler.run()

    plot_1d_histograms(samples)
    plot_2d_scatter(samples)
    plot_3d_chain(samples)

if __name__ == "__main__":
    main()
