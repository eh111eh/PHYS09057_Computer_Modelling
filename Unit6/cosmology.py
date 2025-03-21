import astropy.cosmology
import numpy as np
from scipy.interpolate import interp1d

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
        self._astropy_cosmo = astropy.cosmology.LambdaCDM(H0=H0, Om0=Omega_m, Ode0=Omega_lambda)

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

    def distance_modulus(self, z_values):
        """
        Calculate the distance modulus using astropy's LambdaCDM model
        (Inspired by fallback_cosmology.py file on Learn).

        Parameters:
            z_max (float): Maximum redshift used for distance calculation
            n (int): Number of points used for trapezoidal integration
            z_values (array-like): Redshift values to compute the distance modulus

        Returns:
            np.ndarray: The distance modulus μ(z) (unitless).
        """
        return self._astropy_cosmo.distmod(z_values).value
    
def test_cosmology():
    # use the three test cosmologies given in the notes
    cosmo1 = Cosmology(60., 0.25, 0.7)
    cosmo2 = Cosmology(70., 0.3, 0.7)
    cosmo3 = Cosmology(80., 0.4, 0.65)

    # The three redshifts given in the notes
    mu1 = cosmo1.distance_modulus(0.5)
    mu2 = cosmo2.distance_modulus(1.0)
    mu3 = cosmo3.distance_modulus(1.5)

    # These are the values I get on my machine with my version of astropy
    expected1 = 42.615802574441574
    expected2 = 44.10023765554372
    expected3 = 44.76602040622335

    # The smallest error bar in our supernova data
    sigma = 0.2

    # The fractional difference from the expected values.
    # We don't expect the values to be absolutely identical between computers
    diff1 = abs(expected1 - mu1) / sigma
    diff2 = abs(expected2 - mu2) / sigma
    diff3 = abs(expected3 - mu3) / sigma

    # we will print a friendly message explaining whether or not the above differences are fine
    okay1 = "fine" if diff1 < 0.02 else "not good enough - please report this to Joe"
    okay2 = "fine" if diff2 < 0.02 else "not good enough - please report this to Joe"
    okay3 = "fine" if diff3 < 0.02 else "not good enough - please report this to Joe"

    print(f"mu1 = {mu1} - difference from expected is {diff1:f}% of error bar, which is {okay1}")
    print(f"mu1 = {mu2} - difference from expected is {diff2:f}% of error bar, which is {okay2}")
    print(f"mu1 = {mu3} - difference from expected is {diff3:f}% of error bar, which is {okay3}")




if __name__ == '__main__':
    test_cosmology()
