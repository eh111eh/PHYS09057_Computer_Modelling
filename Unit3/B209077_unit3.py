import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.integrate as integrate

# ------------------ Task 4.1 Adding numerical integration methods ------------------

class Cosmology:
    def __init__(self, H0, Omega_m, Omega_lambda):
        """
        This class represents a cosmological model with parameters H0,
        Omega_m, Omega_lambda, and Omega_k.

        It computes cosmological distances using numerical integration methods
        such as the Rectangle rule, Trapezoid rule, and Simpson's rule.
        """
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_lambda = Omega_lambda
        self.Omega_k = 1 - Omega_m - Omega_lambda

    def integrand(self, z):
        """
        Compute the integrand of the distance formula at a given redshift z:
        [Omega_m * (1 + z)^3 + Omega_k * (1 + z)^2 + Omega_lambda]^(-1/2)
        """
        return (self.Omega_m * (1 + z)**3 + self.Omega_k * (1 + z)**2 + self.Omega_lambda)**(-0.5)

    def distance_rectangle_rule(self, z, n):
        """
        Compute the cosmological distance using the Rectangle rule for integration.
        It approximates the integral using the rectangle rule, where the integrand
        is evaluated at equally spaced intervals, and the total area is approximated by summing
        the areas of rectangles under the curve. Due to c [km/s] and H0 [km/s/Mpc],
        the resulted distance is in Megaparsecs (Mpc). Here, since the speed of light,
        c, is a constant and does not change its value, so let's set it as a numeric
        value '3e5 [km/s]'.

        Parameters:
            z (float): The upper limit of the integration (redshift)
            n (int): Number of points for the numerical integration

        Returns:
            float: The cosmological distance in Megaparsecs [Mpc]
        """
        z_values = np.linspace(0, z, n)
        dz = z / n
        integral = np.sum([self.integrand(z_i) for z_i in z_values]) * dz
        return (3e5 / self.H0) * integral

    def distance_trapezoid_rule(self, z, n):
        """
        Compute the cosmological distance using the Trapezoid rule for numerical integration.
        It approximates the integral by summing the areas of trapezoids formed
        between each pair of consecutive points.

        Parameters:
            z (float): The upper limit of the integration (redshift)
            n (int): Number of points for the numerical integration

        Returns:
            float: The cosmological distance in Megaparsecs [Mpc]
        """
        z_values = np.linspace(0, z, n)
        dz = z / (n - 1)
        integral = (0.5 * self.integrand(z_values[0]) +
                    np.sum([self.integrand(z_i) for z_i in z_values[1:-1]]) +
                    0.5 * self.integrand(z_values[-1])) * dz
        return (3e5 / self.H0) * integral

    def distance_simpsons_rule(self, z, n):
        """
        Compute the cosmological distance using Simpson's rule for numerical integration.
        It approximates the integral by fitting a quadratic curve
        between each pair of points, and summing the areas under these curves.
        The number of intervals must be odd for Simpson's rule to work properly.

        Parameters:
            z (float): The upper limit of the integration (redshift)
            n (int): Number of points for the numerical integration (must be odd)

        Returns:
            float: The cosmological distance in Megaparsecs [Mpc]
        """
        if n % 2 == 0:
            n += 1  # Odd number of intervals
        z_values = np.linspace(0, z, n)
        dz = z / (n - 1)
        integral = (self.integrand(z_values[0]) + self.integrand(z_values[-1]) +
                    4 * np.sum([self.integrand(z_values[i]) for i in range(1, n - 1, 2)]) +
                    2 * np.sum([self.integrand(z_values[i]) for i in range(2, n - 2, 2)])) * dz / 3
        return (3e5 / self.H0) * integral

    def cumulative_trapezoid(self, z_max, n):
        """
        Compute cumulative distances to a range of redshifts using the Trapezoid rule.
        It calculates the distance iteratively for increasing redshift values
        using the trapezoid rule, and returns an array of cumulative distances.

        Parameters:
            z_max (float): The maximum redshift value
            n (int): Number of points for the numerical integration
        
        Returns:
            np.ndarray: Array of cumulative distances from redshift 0 to z_max in Megaparsecs [Mpc]
        """
        z_values = np.linspace(0, z_max, n)
        dz = z_max / (n - 1)
        distances = np.zeros(n)
        for i in range(1, n):
            distances[i] = distances[i-1] + 0.5 * dz * (self.integrand(z_values[i-1]) + self.integrand(z_values[i]))
        return (3e5 / self.H0) * distances
    
    def interpolate_distances(self, z_max, n, z_values):
        """
        Interpolate distances using SciPy's interpolation method based on cumulative trapezoid results.

        Parameters:
            z_max (float): Maximum redshift used in the interpolation
            n (int): Number of points used for trapezoidal integration
            z_values (array-like): Redshift values to interpolate
        
        Returns:
            np.ndarray: Interpolated distances [Mpc]
        """
        z_grid = np.linspace(0, z_max, n)
        distances = self.cumulative_trapezoid(z_max, n)
        interpolator = interp1d(z_grid, distances, kind='cubic', fill_value="extrapolate")
        return interpolator(z_values)

    def distance_modulus(self, z_max, n, z_values):
        """
        Calculate the distance modulus using the interpolated distances:
        mu = 5 * log10(D(z) / 10)

        where D(z) is the distance in parsecs, so the distances are converted to parsecs.
        The result μ(z) is unitless as it represents a logarithmic measure.

        Parameters:
            z_max (float): Maximum redshift used for distance calculation
            n (int): Number of points used for trapezoidal integration
            z_values (array-like): Redshift values to compute the distance modulus

        Returns:
            np.ndarray: The distance modulus μ(z) (unitless).
        """
        distances = self.interpolate_distances(z_max, n, z_values)
        distances[distances == 0] = np.nan
        mu = 5 * np.log10(distances / 10)
        return mu

# ------------------ Task 4.2 Convergence testing ------------------

def convergence_test():
    """
    Test the convergence of the Rectangle, Trapezoid, and Simpson's rules by comparing
    the results for different numbers of integration points to a highly precise estimate.

    This function plots the absolute fractional error in the distance to redshift z=1
    as a function of the number of integration points for each method.
    """
    model = Cosmology(H0=72.0, Omega_m=0.3, Omega_lambda=0.7)
    z = 1.0
    n_large = 10000
    exact_distance = model.distance_simpsons_rule(z, n_large)

    # Target accuracy based on physical reasoning
    target_accuracy = 10**-6

    # Range of points for testing
    n_values = np.logspace(1, 4, 10, dtype=int)
    rectangle_errors = []
    trapezoid_errors = []
    simpson_errors = []

    for n in n_values:
        rectangle_distance = model.distance_rectangle_rule(z, n)
        trapezoid_distance = model.distance_trapezoid_rule(z, n)
        simpson_distance = model.distance_simpsons_rule(z, n)

        rectangle_errors.append(np.abs((rectangle_distance - exact_distance) / exact_distance))
        trapezoid_errors.append(np.abs((trapezoid_distance - exact_distance) / exact_distance))
        simpson_errors.append(np.abs((simpson_distance - exact_distance) / exact_distance))

    plt.figure()
    plt.loglog(n_values, rectangle_errors, label='Rectangle Rule')
    plt.loglog(n_values, trapezoid_errors, label='Trapezoid Rule')
    plt.loglog(n_values, simpson_errors, label="Simpson's Rule")
    
    # Plot the target accuracy line
    plt.axhline(y=target_accuracy, color='r', linestyle='--', label=f'Target Accuracy: {target_accuracy}')
    
    plt.xlabel('Number of Points')
    plt.ylabel('Absolute Fractional Error')
    plt.title('Convergence of Numerical Integration Methods')
    plt.legend()
    plt.grid(True)
    plt.show()

# ------------------ Task 4.3 Cumulative version ------------------

def test_cumulative_trapezoid():
    model = Cosmology(H0=70.0, Omega_m=0.3, Omega_lambda=0.7)
    z_max = 1.0
    n = 100
    distances = model.cumulative_trapezoid(z_max, n)

    z_values = np.linspace(0, z_max, n)
    plt.figure()
    plt.plot(z_values, distances)
    plt.xlabel('Redshift z')
    plt.ylabel('Distance (Mpc)')
    plt.title('Cumulative Distance to Redshift')
    plt.grid(True)
    plt.show()

# ------------------ Task 4.4 Exploration ------------------

def explore_parameters():
    """
    Explore the effect of changing H0, Omega_m, and Omega_lambda on the distance modulus mu(z).
    Produce three separate plots for each parameter change.
    """
    z_values = np.linspace(0, 1.0, 100)

    # Plot 1: Varying H0 while keeping Omega_m and Omega_lambda constant
    plt.figure()
    for H0 in [60, 70, 80]:
        model = Cosmology(H0=H0, Omega_m=0.3, Omega_lambda=0.7)
        mu = model.distance_modulus(1.0, 1000, z_values)
        plt.plot(z_values, mu, label=f'H0 = {H0}')
    plt.xlabel('Redshift z')
    plt.ylabel('Distance Modulus mu(z)')
    plt.title("Effect of Varying H0 on Distance Modulus")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 2: Varying Omega_m while keeping H0 and Omega_lambda constant
    plt.figure()
    for Omega_m in [0.2, 0.3, 0.4]:
        model = Cosmology(H0=70, Omega_m=Omega_m, Omega_lambda=0.7)
        mu = model.distance_modulus(1.0, 1000, z_values)
        plt.plot(z_values, mu, label=f'Omega_m = {Omega_m}')
    plt.xlabel('Redshift z')
    plt.ylabel('Distance Modulus mu(z)')
    plt.title("Effect of Varying Omega_m on Distance Modulus")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 3: Varying Omega_lambda while keeping H0 and Omega_m constant
    plt.figure()
    for Omega_lambda in [0.6, 0.7, 0.8]:
        model = Cosmology(H0=70, Omega_m=0.3, Omega_lambda=Omega_lambda)
        mu = model.distance_modulus(1.0, 1000, z_values)
        plt.plot(z_values, mu, label=f'Omega_lambda = {Omega_lambda}')
    plt.xlabel('Redshift z')
    plt.ylabel('Distance Modulus mu(z)')
    plt.title("Effect of Varying Omega_lambda on Distance Modulus")
    plt.legend()
    plt.grid(True)
    plt.show()

# ------------------ Additional test function ------------------

def check_scipy_custom():
    """
    Additional function to compare results from our custom functions and scipy libraries.
    """
    model = Cosmology(H0=72.0, Omega_m=0.3, Omega_lambda=0.7)
    z = 1.0
    n = 10000

    # Define the integrand function using model parameters
    integrand_function = lambda z: model.integrand(z)

    # Scipy integration results
    rect_scipy = integrate.fixed_quad(integrand_function, 0, z, n=n)[0]
    trap_scipy = integrate.quad(integrand_function, 0, z, limit=n)[0]
    simp_scipy = integrate.simpson([integrand_function(z_i) for z_i in np.linspace(0, z, n)], dx=z/n)

    # Convert results to distance in Mpc
    rect_scipy_distance = (3e5 / model.H0) * rect_scipy
    trap_scipy_distance = (3e5 / model.H0) * trap_scipy
    simp_scipy_distance = (3e5 / model.H0) * simp_scipy

    # Calculate distances using our custom functions
    rect_distance = model.distance_rectangle_rule(z, n)
    trap_distance = model.distance_trapezoid_rule(z, n)
    simp_distance = model.distance_simpsons_rule(z, n)

    print(f"Rectangle Rule Distance (Custom): {rect_distance:.5f} Mpc")
    print(f"Rectangle Rule Distance (SciPy): {rect_scipy_distance:.5f} Mpc")
    print(f"Difference: {abs(rect_distance - rect_scipy_distance):.5f} Mpc\n")

    print(f"Trapezoid Rule Distance (Custom): {trap_distance:.5f} Mpc")
    print(f"Trapezoid Rule Distance (SciPy): {trap_scipy_distance:.5f} Mpc")
    print(f"Difference: {abs(trap_distance - trap_scipy_distance):.5f} Mpc\n")

    print(f"Simpson's Rule Distance (Custom): {simp_distance:.5f} Mpc")
    print(f"Simpson's Rule Distance (SciPy): {simp_scipy_distance:.5f} Mpc")
    print(f"Difference: {abs(simp_distance - simp_scipy_distance):.5f} Mpc\n")

    # Cumulative trapezoid validation
    z_values = np.linspace(0, z, n)
    custom_cumulative_distances = model.cumulative_trapezoid(z, n)

    # SciPy cumulative trapezoid integration
    cumulative_trap_scipy = integrate.cumulative_trapezoid([model.integrand(z_i) for z_i in z_values], z_values, initial=0)
    cumulative_trap_scipy_distances = (3e5 / model.H0) * cumulative_trap_scipy

    # Print difference between custom and scipy cumulative trapezoid results
    print("\nCumulative Trapezoid Rule Validation:")
    for i in range(0, n, n // 10):
        print(f"z={z_values[i]:.2f}: Custom = {custom_cumulative_distances[i]:.5f} Mpc, "
              f"SciPy = {cumulative_trap_scipy_distances[i]:.5f} Mpc, "
              f"Difference = {abs(custom_cumulative_distances[i] - cumulative_trap_scipy_distances[i]):.5f} Mpc")

def main():
    """
    Run the following:
        i. Tests on the three numerical integration methods (Rectangle, Trapezoid, Simpson)
           to check if they produce the correct cosmological distance for z = 1.
           The expected distance is approximately 3200 Mpc for the default cosmological parameters.
        ii. The convergence test and the cumulative trapezoid test.
        iii. Explorational function to explore the effect of changing parameters.
        iv. Additional test function to compare scipy libraries and our custom functions.
    """
    model = Cosmology(H0=72.0, Omega_m=0.3, Omega_lambda=0.7)
    z = 1.0
    n = 1000

    # Test the three integration methods
    dist_rect = model.distance_rectangle_rule(z, n)
    dist_trap = model.distance_trapezoid_rule(z, n)
    dist_simp = model.distance_simpsons_rule(z, n)

    print(f"Distance (Rectangle Rule): {dist_rect:.2f} Mpc")
    print(f"Distance (Trapezoid Rule): {dist_trap:.2f} Mpc")
    print(f"Distance (Simpson's Rule): {dist_simp:.2f} Mpc")

    # Check if they are close to the expected value ~3200 Mpc
    expected_distance = 3200
    assert np.isclose(dist_rect, expected_distance, rtol=0.1), "Rectangle rule result is too far from the expected value"
    assert np.isclose(dist_trap, expected_distance, rtol=0.1), "Trapezoid rule result is too far from the expected value"
    assert np.isclose(dist_simp, expected_distance, rtol=0.1), "Simpson's rule result is too far from the expected value"

    # The convergence test
    convergence_test()

    # The cumulative trapezoid test
    test_cumulative_trapezoid()

    # The effect of changing parameters
    explore_parameters()

    # Additional test function to compare scipy libraries and custom functions
    check_scipy_custom()

if __name__ == "__main__":
    main()
