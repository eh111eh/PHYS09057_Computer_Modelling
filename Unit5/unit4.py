import numpy as np
from cosmology import Cosmology
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution

# ------------------ Task 4.1 Likelihoods ------------------

class Likelihood:
    def __init__(self, data_file, M=-19.3, N=1000):
        """
        Initialise the Likelihood class by loading the supernova data.

        Parameters
        ==========
        data_file : str
            Path to the file containing the supernova data.
        M : float
            The absolute magnitude of Type Ia supernovae. Default is -19.3.
        N : int
            Number of integration points. Default is 1000.
        """
        data = np.loadtxt(data_file, unpack=True)
        self.z = data[0]         # Redshift values
        self.mu_obs = data[1]    # Observed apparent magnitudes
        self.mu_err = data[2]    # Errors on apparent magnitudes
        self.M = M               # Absolute magnitude
        self.N = N               # Number of integration points

    def model_distance_modulus(self, z, theta):
        """
        Compute the theoretical distance modulus for the given redshifts
        using the Cosmology class from cosmology.py.

        Parameters
        ==========
        z : np.ndarray
            Array of redshifts.
        theta : list or np.ndarray
            Cosmological parameters [Omega_m, Omega_Lambda, H0].
        """
        Omega_m, Omega_Lambda, H0 = theta
        cosmo = Cosmology(H0, Omega_m, Omega_Lambda)

        # Compute cumulative luminosity distance in Mpc
        cumulative_distances = cosmo.cumulative_trapezoid(np.max(z), self.N)

        # Interpolate distances for given redshift values
        interpolated_distances = np.interp(z, np.linspace(0, np.max(z), self.N), cumulative_distances)

        """# Debugging: Print interpolated distances
        test_z = np.array([0.1, 0.5, 1.0, 1.5])
        test_distances = cosmo.interpolate_distances(2.0, 1000, test_z)
        print(f"Interpolated distances at {test_z}: {test_distances}")"""

        # Apply the (1+z) scaling for luminosity distance
        D_L_mpc = interpolated_distances * (1 + z)

        # Convert to parsecs and compute distance modulus
        D_L_pc = D_L_mpc * 1e6  # Convert Mpc to parsecs
        mu_model = 5 * np.log10(D_L_pc / 10)

        return mu_model + self.M

    def __call__(self, theta, model="standard"):
        """
        Compute the log-likelihood for the given cosmological parameters.

        Parameters
        ==========
        theta : list or np.ndarray
            Cosmological parameters [Omega_m, Omega_Lambda, H0].
        model : str
            Model choice ("standard" or "no_omega_lambda"). Default is "standard".
        """
        if model == "no_omega_lambda":
            # Fix Omega_Lambda to 0
            theta = [theta[0], 0.0, theta[1]]

        mu_model = self.model_distance_modulus(self.z, theta)
        residuals = (self.mu_obs - mu_model) / self.mu_err
        chi2 = np.sum(residuals**2)
        if chi2 > 1e6:
            return -1e6
        return -0.5 * chi2

def test_likelihood():
    """
    Test the Likelihood class using the pantheon_data.txt file and
    compute the log-likelihood for given cosmological parameters.
    Also, test the convergence of the likelihood calculation with respect to N.
    """
    data_file = "pantheon_data.txt"

    # Initialize the Likelihood object
    likelihood = Likelihood(data_file)

    # Cosmological parameters
    theta = [0.3, 0.7, 70.0]

    # Compute the log-likelihood
    log_likelihood = likelihood(theta)
    print(f"Final Log-Likelihood for theta={theta}: {log_likelihood}")

    # Test convergence
    print("\nTesting convergence of log-likelihood with respect to parameter N:")
    N_values = np.arange(100, 2000, 100)
    log_likelihoods = []

    for N in N_values:
        likelihood.N = N  # Update the number of integration points
        log_likelihoods.append(likelihood(theta))

    # Calculate changes in log-likelihood
    deltas = np.abs(np.diff(log_likelihoods))

    # Plot the convergence
    plt.figure(figsize=(10, 6))
    plt.plot(N_values, log_likelihoods, marker="o", label="Log-Likelihood")
    plt.axhline(-476, color="r", linestyle="--", label="Expected Value (-476)")
    plt.xlabel("Parameter N (Number of Integration Points)")
    plt.ylabel("Log-Likelihood")
    plt.title("Convergence of Log-Likelihood with Integration Points")
    plt.grid()
    plt.legend()
    plt.show()

    # Print convergence results
    print("\nConvergence Results:")
    for i in range(len(deltas)):
        print(f"N = {N_values[i]}, Log-Likelihood = {log_likelihoods[i]}, Change = {deltas[i]}")

    print(f"\nFinal Log-Likelihood for theta={theta}: {log_likelihoods[-1]}")

# ------------------ Task 4.2 Optimization ------------------

def optimize_likelihood(data_file, initial_guess, bounds, method="L-BFGS-B", M=-19.3):
    """
    Optimise the likelihood to find the best-fitting parameters.

    Parameters
    ==========
    initial_guess : list or np.ndarray
        Initial guess for the parameters [Omega_m, Omega_Lambda, H0].
    bounds : list of tuples
        Bounds for the parameters [(min1, max1), (min2, max2), ...].
    method : str
        Optimisation method (default: "L-BFGS-B").
    M : float
        Absolute magnitude of supernovae (default: -19.3).

    Returns
    =======
    dict
        Dictionary containing the optimization results, best-fit parameters,
        and log-likelihood value.
    """
    likelihood = Likelihood(data_file, M=M)

    # Optimisation for log-likelihood
    result = minimize(likelihood, initial_guess, bounds=bounds, method=method, options={"disp": True})

    # Extract the best-fitting parameters and log-likelihood value
    best_fit_params = result.x
    log_likelihood = -result.fun

    return {
        "result": result,
        "best_fit_params": best_fit_params,
        "log_likelihood": log_likelihood
    }

def optimize_with_differential_evolution(data_file, bounds, M=-19.3):
    """
    Optimise the likelihood using the global method Differential Evolution.

    Parameters
    ==========
    bounds : list of tuples
        Bounds for the parameters [(min1, max1), (min2, max2), ...].
    M : float
        Absolute magnitude of supernovae.

    Returns
    =======
    dict
        Dictionary containing the optimisation results, best-fit parameters,
        and log-likelihood value.
    """
    likelihood = Likelihood(data_file, M=M)

    # Minimise the negative log-likelihood using Differential Evolution
    result = differential_evolution(likelihood, bounds)

    # Extract the best-fitting parameters and log-likelihood value
    best_fit_params = result.x
    log_likelihood = -result.fun

    return {
        "result": result,
        "best_fit_params": best_fit_params,
        "log_likelihood": log_likelihood
    }

# ------------------ Task 4.3 Changing the model ------------------

def optimize_omega_lambda_zero(data_file, bounds, M=-19.3):
    """
    Optimise the likelihood for the model where Omega_Lambda is fixed to 0.

    Parameters
    ==========
    data_file : str
        Path to the file containing the supernova data.
    bounds : list of tuples
        Bounds for the parameters [(min1, max1), (min2, max2)].
        Only Omega_m and H0 are optimized (Omega_Lambda = 0).
    M : float
        Absolute magnitude of supernovae.

    Returns
    =======
    dict
        Dictionary containing the optimization results, best-fit parameters,
        and log-likelihood value.
    """
    likelihood = Likelihood(data_file, M=M)

    # Optimisation for log-likelihood using Differential Evolution
    result = differential_evolution(
        lambda params: likelihood([params[0], params[1]], model="no_omega_lambda"),
        bounds
    )

    # Extract the best-fitting parameters and log-likelihood value
    best_fit_params = [result.x[0], 0.0, result.x[1]]  # Add Omega_Lambda = 0
    log_likelihood = -result.fun

    return {
        "result": result,
        "best_fit_params": best_fit_params,
        "log_likelihood": log_likelihood
    }

def compare_models_using_bic(standard_result, omega_lambda_zero_result, num_data_points):
    """
    Compare the standard model and the Omega_Lambda = 0 model using BIC.

    Parameters
    ==========
    standard_result : dict
        Optimization result for the standard model.
    omega_lambda_zero_result : dict
        Optimization result for the model with Omega_Lambda = 0.
    num_data_points : int
        The number of data points in the dataset.

    Returns
    =======
    None
        Prints the BIC values for both models and determines which fits the data better.
    """
    # Extract log-likelihoods and number of parameters
    standard_log_likelihood = standard_result["log_likelihood"]
    omega_lambda_zero_log_likelihood = omega_lambda_zero_result["log_likelihood"]

    # Number of parameters
    standard_num_params = 3  # Omega_m, Omega_Lambda, H0
    omega_lambda_zero_num_params = 2  # Omega_m, H0

    # Compute BIC for both models
    standard_bic = standard_num_params * np.log(num_data_points) - 2 * standard_log_likelihood
    omega_lambda_zero_bic = omega_lambda_zero_num_params * np.log(num_data_points) - 2 * omega_lambda_zero_log_likelihood

    # Print results
    print("\nModel Comparison Using BIC:")
    print(f"Standard Model - Log-Likelihood: {standard_log_likelihood}, BIC: {standard_bic}")
    print(f"Omega_Lambda = 0 Model - Log-Likelihood: {omega_lambda_zero_log_likelihood}, BIC: {omega_lambda_zero_bic}")

    # Determine which model fits better
    if standard_bic < omega_lambda_zero_bic:
        print("Based on BIC, the standard model fits the data better.")
    else:
        print("Based on BIC, the model with Omega_Lambda = 0 fits the data better.")

def plot_results(data_file, best_fit_params, M=-19.3):
    """
    Generate plots for the best-fit model and residuals.

    Parameters
    ==========
    best_fit_params : list or np.ndarray
        Best-fitting parameters [Omega_m, Omega_Lambda, H0].
    M : float
        Absolute magnitude of supernovae.
    """
    likelihood = Likelihood(data_file, M=M)
    z = likelihood.z
    mu_obs = likelihood.mu_obs
    mu_err = likelihood.mu_err

    # Compute the model predictions
    mu_model = likelihood.model_distance_modulus(z, best_fit_params)

    # Plot data with error bars and best-fit model
    plt.figure(figsize=(10, 6))
    plt.errorbar(z, mu_obs, yerr=mu_err, fmt='o', label='Observed Data')
    plt.plot(z, mu_model, label='Best-Fit Model', color='red')
    plt.xlabel("Redshift (z)")
    plt.ylabel("Distance Modulus (mu)")
    plt.title("Observed Data and Best-Fit Model")
    plt.legend()
    plt.grid()
    plt.savefig("best_fit_model.png")
    plt.show()

    # Compute and plot residuals
    residuals = (mu_obs - mu_model) / mu_err
    plt.figure(figsize=(10, 6))
    plt.plot(z, residuals, 'o', label='Residuals')
    plt.axhline(0, color='red', linestyle='--', label='Zero Residual Line')
    plt.xlabel("Redshift (z)")
    plt.ylabel("Residuals")
    plt.title("Residuals of Observed Data vs Best-Fit Model")
    plt.legend()
    plt.grid()
    plt.savefig("residuals.png")
    plt.show()

    # Print residual statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    print(f"Mean of residuals: {mean_residual}")
    print(f"Standard deviation of residuals: {std_residual}")


if __name__ == '__main__':
    # Task 4.1: Test Likelihood
    test_likelihood()

    # Task 4.2: Optimisation
    data_file = "pantheon_data.txt"
    bounds = [(0.1, 0.4), (0.6, 1.0), (60, 80)]
    initial_guess = [0.3, 0.7, 70.0]

    # Perform global optimisation for the standard model using Differential Evolution
    global_result = optimize_with_differential_evolution(data_file, bounds, M=-19.3)
    print("\nStandard Model Optimization:")
    print("\nGlobal Optimization (Differential Evolution):")
    print(f"Best-Fit Parameters: {global_result['best_fit_params']}, Log-Likelihood: {global_result['log_likelihood']}")

    # Optimisation for Omega_Lambda = 0
    omega_lambda_zero_bounds = [(0.1, 0.4), (60, 80)]  # Only Omega_m and H0 bounds
    omega_lambda_zero_result = optimize_omega_lambda_zero(data_file, omega_lambda_zero_bounds, M=-19.3)
    print("\nOptimization for Omega_Lambda = 0:")
    print(f"Best-Fit Parameters: {omega_lambda_zero_result['best_fit_params']}")
    print(f"Log-Likelihood: {omega_lambda_zero_result['log_likelihood']}")

    # Plot results for both models
    plot_results(data_file, global_result['best_fit_params'], M=-19.3)
    plot_results(data_file, omega_lambda_zero_result['best_fit_params'], M=-19.3)

    # Compare models
    data = np.loadtxt("pantheon_data.txt", unpack=True)
    num_data_points = len(data[0])
    compare_models_using_bic(global_result, omega_lambda_zero_result, num_data_points)