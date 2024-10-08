import numpy as np
import matplotlib.pyplot as plt

# ------------------ Task 4.1 Creating numpy arrays ------------------

def task1a(m, n):
    """
    This function generates three numpy arrays based on inputs m and n.

    Parameters:
        m (int): Size for the first array and number of rows for the third array.
        n (int): Size for the second array and number of columns for the third array.

    Returns a tuple containing:
        - A 1D array of all zeros of size m.
        - A 1D array containing the numbers 1, 2, 3, … n.
        - A 2D array of shape (m, n) with random uniform numbers in the range [0, 1).
    """
    arr1 = np.zeros(m)
    arr2 = np.arange(1, n + 1)
    arr3 = np.random.uniform(0, 1, (m, n))
    return arr1, arr2, arr3

def task1c(m, n):
    """
    This function calls task1a and calculates the mean of the second array and the max of the third array.

    Parameters:
        m (int): Size for the first array and number of rows for the third array.
        n (int): Size for the second array and number of columns for the third array.

    Returns a tuple containing
        - Mean of the second array.
        - Max value of the third array.
    """
    _, arr2, arr3 = task1a(m, n)
    mean_arr2 = np.mean(arr2)
    max_arr3 = np.max(arr3)
    return mean_arr2, max_arr3

def task1d(a):
    """
    This function modifies the input array by squaring each element.

    Parameters:
        a (numpy.ndarray): The input array to modify.

    Returns None.
    """
    # Explanation: Mutable objects like numpy arrays can be changed in place, while immutable objects like integers and strings cannot.
    # This means changes to a mutable object will affect the original object, whereas changes to an immutable object will create a new one.
    a[:] = a ** 2

# ------------------ Task 4.2 Vector arithmetic ------------------

def task2a(a, b, t):
    """
    This function computes 2t(a + b) for numpy vectors a and b and scalar t.

    Parameters:
        a (numpy.ndarray): The first vector.
        b (numpy.ndarray): The second vector.
        t (float): A scalar value.

    Returns numpy.ndarray as the result of the computation.
    """
    return 2 * t * (a + b)

def task2b(x, y):
    """
    This function computes the Euclidean distance between two 3D position vectors x and y.

    Parameters:
        x (numpy.ndarray): The first position vector.
        y (numpy.ndarray): The second position vector.

    Returns float: The distance between x and y.

    I believe the hint refers to the 'numpy.linalg.norm' function, which can compute the
    Euclidean distance (norm/magnitude of a vector) between two vectors.
    """
    return np.linalg.norm(x - y)

def task3a(v1, v2):
    """
    This function computes v1 x v2 and -v2 x v1 which demonstrates the symmetry of
    vector cross product.

    Parameters:
        v1 (numpy.ndarray): First vector.
        v2 (numpy.ndarray): Second vector.

    Returns tuple: Results of v1 x v2 and -v2 x v1.
    """
    return np.cross(v1, v2), -np.cross(v2, v1)

def task3b(v1, v2, v3):
    """
    This function verifies the distributive property of the cross product: v1 x (v2 + v3).

    Parameters:
        v1, v2, v3 (numpy.ndarray): Input vectors.

    Returns tuple: Results of both sides of the distributive property.
    """
    return np.cross(v1, v2 + v3), np.cross(v1, v2) + np.cross(v1, v3)

def task3c(v1, v2, v3):
    """
    This function verifies the scalar triple product identity: v1 x (v2 x v3).
    
    Parameters:
        v1, v2, v3 (numpy.ndarray): Input vectors.

    Returns tuple: Results of both sides of the identity.
    """
    return np.cross(v1, np.cross(v2, v3)), (np.dot(v1, v3) * v2 - np.dot(v1, v2) * v3)

# ------------------ Task 4.3 2D Arrays ------------------

def task4a(n):
    """
    This function creates a square matrix M of size n x n where M_ij = i + 2j.
    
    Parameters:
        n (int): The size of the matrix.

    Returns numpy.ndarray: The created matrix.
    """
    return np.fromfunction(lambda i, j: i + 2 * j, (n, n), dtype=int)

def task4b(n):
    """
    This function computes a 1D array yi = sum(M_ij) for j in range(0, n).
    
    Parameters:
        n (int): Size of the square matrix.
    
    Returns numpy.ndarray: The computed 1D array.
    """
    M = task4a(n)
    return np.sum(M, axis=1)

# ------------------ Task 4.4 Likelihoods ------------------

def task5(d, mu, sigma):
    """
    This function computes the Gaussian log-likelihood given three vectors.

    Parameters:
        d (numpy.ndarray): Data points.
        mu (numpy.ndarray): Mean values.
        sigma (numpy.ndarray): Standard deviations.

    Returns float: The log-likelihood.
    """
    return -0.5 * np.sum(((d - mu) / sigma) ** 2)

# ------------------ Task 4.5 Data analysis ------------------

def task6():
    """
    This function performs basic data analysis on the given file, 'data.txt',
    and displays means and standard deviations, histograms of individual data points,
    and scatter plots of pairs of data points.

    Parameters:
        filename (str): The path to the data file.

    Returns None.
    """
    # Load the data from the text file
    data = np.loadtxt('data.txt')

    # Calculate mean and standard deviation for each column
    means = np.mean(data, axis=0)
    std_devs = np.std(data, axis=0)

    # Display the results on screen
    print("Data Analysis Results:")
    print(f"Means for columns (omegamh2, omegabh2, H0): {means}")
    print(f"Standard Deviations for columns (omegamh2, omegabh2, H0): {std_devs}")

    column_names = ['omegamh2', 'omegabh2', 'H0']

    # Generate histograms for each column
    for i, col_name in enumerate(column_names):
        plt.figure(figsize=(6, 4))
        plt.hist(data[:, i], bins=30, alpha=0.7, edgecolor='black')
        plt.title(f'Histogram of {col_name}')
        plt.xlabel(col_name)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    # Generate scatter plots for each pair of columns
    pairs = [(0, 1), (0, 2), (1, 2)]  # (omegamh2, omegabh2), (omegamh2, H0), (omegabh2, H0)
    for i, (x_idx, y_idx) in enumerate(pairs):
        plt.figure(figsize=(6, 4))
        plt.scatter(data[:, x_idx], data[:, y_idx], alpha=0.7)
        plt.title(f'Scatter Plot of {column_names[x_idx]} vs {column_names[y_idx]}')
        plt.xlabel(column_names[x_idx])
        plt.ylabel(column_names[y_idx])
        plt.grid(True)
        plt.show()

# ------------------ Task 4.6 Main function ------------------

def main():
    """
    Main function demonstrating the use of other functions.
    """
    m, n = 5, 3
    print("Task 1a:", task1a(m, n))
    print("Task 1c:", task1c(m, n))
    
    arr = np.array([1, 2, 3])
    task1d(arr)
    print("Task 1d (squared array):", arr)
    
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    t = 2
    print("Task 2a:", task2a(a, b, t))
    
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    print("Task 2b:", task2b(x, y))
    
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    v3 = np.array([7, 8, 9])
    print("Task 3a:", task3a(v1, v2))
    print("Task 3b:", task3b(v1, v2, v3))
    print("Task 3c:", task3c(v1, v2, v3))
    
    n = 4
    print("Task 4a:", task4a(n))
    print("Task 4b:", task4b(n))
    
    d = np.array([1, 2, 3])
    mu = np.array([1, 1, 1])
    sigma = np.array([1, 1, 1])
    print("Task 5:", task5(d, mu, sigma))
    
    task6()

if __name__ == "__main__":
    main()