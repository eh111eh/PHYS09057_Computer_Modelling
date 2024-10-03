import numpy as np
import matplotlib.pyplot as plt

# Define the Cosmology class
class Cosmology:
    def __init__(self, H0, Omega_m, Omega_lambda):
        """
        The __init__ method initializes the object with specific cosmological
        parameters—H0, Omega_m, and Omega_lambda—and calculates Omega_k as
        1 - Omega_m - Omega_lambda.

        The self keyword refers to the instance of the class.
        It allows access to the class’s variables and methods and sets the
        values of H0, Omega_m, Omega_lambda, and Omega_k as attributes of
        the specific instance of the Cosmology class being created.
        """
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_lambda = Omega_lambda
        self.Omega_k = 1 - Omega_m - Omega_lambda

    def integrand(self, z):
        """
        Compute the integrand of the distance formula:
        [Omega_m * (1 + z')^3 + Omega_k * (1 + z')^2 + Omega_lambda]^(-1/2)
        """
        return (self.Omega_m * (1 + z)**3 + self.Omega_k * (1 + z)**2 + self.Omega_lambda)**(-0.5)

    def distance(self, z):
        """
        Calculate the distance D(z) to a galaxy using the redshift z
        by integrating the distance formula.
        """
        c = 3e5  # speed of light in km/s
        return c / self.H0 * np.array([self.integrand(z_prime) for z_prime in z]).sum() * (z[1] - z[0])

    def is_flat(self):
        """
        Return whether the universe is flat, i.e., Omega_k is close to zero.
        """
        return np.isclose(self.Omega_k, 0)

    def set_Omega_m(self, new_Omega_m):
        """
        Set Omega_m and modify Omega_lambda accordingly
        to keep the curvature of the Universe the same.
        """
        self.Omega_m = new_Omega_m
        self.Omega_lambda = 1 - self.Omega_m - self.Omega_k

    def set_Omega_lambda(self, new_Omega_lambda):
        """
        Set Omega_lambda and modify Omega_m accordingly
        to keep the curvature of the Universe the same.
        """
        self.Omega_lambda = new_Omega_lambda
        self.Omega_m = 1 - self.Omega_lambda - self.Omega_k

    def Omega_m_h2(self):
        """
        Return the quantity Omega_m * h^2, where h = H0/100km/s/Mpc).
        """
        h = self.H0 / 100
        return self.Omega_m * h**2

    def __str__(self):
        """
        The __str__ method Return a string describing the model, e.g.,
        <Cosmology with H0=72.0, Omega_m=0.3, Omega_lambda=0.72, Omega_k=0.02>
        """
        return f"<Cosmology with H0={self.H0}, Omega_m={self.Omega_m}, Omega_lambda={self.Omega_lambda}, Omega_k={self.Omega_k}>"

# Test function to demonstrate the use of the Cosmology class
def main():
    # Create a Cosmology object with default parameters
    model = Cosmology(H0=70.0, Omega_m=0.3, Omega_lambda=0.7)

    # Task 4.3.1 - Plotting the integrand between z = 0 and z = 1
    z_values = np.linspace(0, 1, 100)
    integrand_values = [model.integrand(z) for z in z_values]
    plt.figure()
    plt.plot(z_values, integrand_values)
    plt.xlabel('Redshift z')
    plt.ylabel('Integrand Value')
    plt.title('Integrand of Distance Formula')
    plt.grid(True)
    plt.show()

    # Task 4.3.2 - Show how varying Omega_m changes the integrand behavior
    """
    As Omega_m increases (0.3 to 0.5), the integrand value decreases for a given redshift.
    This shows that higher matter density leads to a slower decrease in the
    integrand, implying a greater contribution of matter to the overall energy
    density of the universe.
    """
    model2 = Cosmology(H0=70.0, Omega_m=0.5, Omega_lambda=0.7)

    integrand_values_changed = [model2.integrand(z) for z in z_values]
    plt.figure()
    plt.plot(z_values, integrand_values, label='Omega_m = 0.3')
    plt.plot(z_values, integrand_values_changed, label='Omega_m = 0.5')
    plt.xlabel('Redshift z')
    plt.ylabel('Integrand Value')
    plt.title('Effect of Changing Omega_m on Integrand')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Task 4.3.3 - Use setter methods to show the effect of varying Omega_m and Omega_lambda
    """
    Unlike the previous plot where only Omega_m was changed independently,
    in this plot, changing Omega_m using the setter method also affects Omega_lambda
    to maintain the relationship Omega_k + Omega_m + Omega_lambda = 1.

    As a result, in this plot, the integrand value decreases steeper than the previous plot.
    """
    model = Cosmology(H0=70.0, Omega_m=0.3, Omega_lambda=0.7)
    model.set_Omega_m(0.5)

    integrand_values_modified = [model.integrand(z) for z in z_values]
    plt.figure()
    plt.plot(z_values, integrand_values, label='Omega_m = 0.3')
    plt.plot(z_values, integrand_values_modified, label='Omega_m = 0.5 (by setter)')
    plt.xlabel('Redshift z')
    plt.ylabel('Integrand Value')
    plt.title('Effect of Changing Both Omega_m and Omega_lambda on Integrand')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Task 4.3.4 - Print out various Cosmology objects
    """
    In the previous sections, tasks 4.1 and 4.2 involve creating a cosmological model
    and analyzing how changes in the parameters affect the universe's behavior. And this
    work extend it by:

    i. Creating multiple models with varying parameters (H0, Omega_m, Omega_lambda).
       This explores how each set of parameters affects the model’s characteristics,
       such as the integrand value.

    ii. Comparing models by calculating the difference in the integrand value at z=1
        between consecutive models. We can check how sensitive the model is to changes
        in these parameters.
    """
    # Create and print different Cosmology objects with varying parameters
    models = [
        Cosmology(70.0, 0.3, 0.7),
        Cosmology(68.0, 0.35, 0.65),
        Cosmology(75.0, 0.4, 0.55)
    ]

    for i, cosmo in enumerate(models):
        print(f"Cosmology Model {i + 1}: {cosmo}")
        # Print how this model's integrand at z=1 differs from the previous model
        if i > 0:
            integrand_previous = models[i-1].integrand(1)
            integrand_current = cosmo.integrand(1)
            difference = integrand_current - integrand_previous
            print(f"Difference in integrand at z=1 from previous model: {difference:.5e}\n")

if __name__ == "__main__":
    main()