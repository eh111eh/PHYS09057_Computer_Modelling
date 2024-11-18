# Table of Contents
1. Course Notes
2. Introductory lecture & Workshop mini-lectures

# Course Notes
## Chapter 3. Beyond SciProg
### 3.1. Basic Python
- Role of `__init__` and `self`:
	```python
	 class A(object):
		def __init__(self):
			self.x = 'Hello'
		def method_a(self, foo):
			print self.x + ' ' + foo
 	```
	The `self` variable represents the instance of the object itself. <br/>
	The `__init__()` method takes arguments and assigns them to properties of the object.

- Default argument values <br/>
	i. Non-default arguments must come before those with default values <br/>
	Problematic example:
  ```python
  def wrong_args(mol=1, press, temp=273.15): # Raise a SyntaxError
		pass
  ```
	Correct example:
  ```python
  def correct_args(press, mol=1, temp=273.15):
		pass
  ```	
	ii. Positional argument must come before any keyword arguments <br/>
	Problematic example:
  ```python
  ideal_volume(mol=0.1, 2) # SyntaxError
  ```
	Correct example:
  ```python
	ideal_volume(0.1, 2) # All positional arguments
	ideal_volume(mol=0.1, press=2) # All keyword arguments
	ideal_volume(0.1, press=2) # Positional before keyword
	```

- String formatting <br/>
	i. % Operator
    ```python
		name = "John"
		age = 25
		print("My name is %s and I am %d years old." % (name, age))
    ```
	ii. str.format()
    ```python
		name = "John"
		age = 25
		print("My name is {} and I am {} years old.".format(name, age))
    ```
    ```python
		name = "John"
		age = 25
		print("""My name is {0} and I am {1} years old.
		That means I was born {1} years ago. And to repeat,
		my name is {0}""".format(name, age))
    ```
	iii. f-string <br/>
    ```python
		name = "John"
		age = 25
		print(f"My name is {name} and I am {age} years old.")
    ```
### 3.2. Numpy
#### 3.2.1. Loading numpy and creating arrays
Numpy: library used for numerical and scientific computing. Arrays in numpy are more efficient and structured than Python lists.
- Use `import numpy as mp` to import numpy.
- Create arrays using `np.array()`.
- All elements in a numpy array must be of the same types, and the type can be explicitly set.
Example:
```python
import numpy as mp
array = np.array([[0, 1], [1.0, 0]], dtype=float)
>>> print(array)
[[0.  1.  ]
[1.  0.  ]]
```

#### 3.2.2. Maths with Arrays
Numpy supports element-wise and matrix mathematical operations directly on arrays.
- Arithmetic operators like `+`, `-`, `*`, `/` are element-wise.
- Use `np.dot()` or `@` for matrix multiplication.
- Numpy includes universal functions like `np.exp()` or `np.log()` for element-wise operations.
Example:
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
>>> print(a + b)
[5, 7, 9]
>>> print(np.dot(a, b))
32 # Dot product
```

#### 3.2.3. Reshaping and Slicing
Arrays can be reshaped or sliced to access specific elements or change their structure.
- Use `.reshape()` to change the shape of an array.
- Use slicing with colons(`:`) to extract parts of an array.
Example:
```python
array = np.arange(12).reshape(3, 4) # reshape into 3 rows and 4 columns
>>> print(array[:, 1])
[1, 5, 9] # Second column
```

#### 3.2.4. Broadcasting
Numpy automatically handles arrays of different shapes during arithmetic if their shapes are compatible.
- Dimensions must match or one must be 1 for broadcasting to occur.
Example:
```python
a = np.array([1, 2, 3])
b = np.array([[1], [2], [3]])
>>> print(a + b)
[[2 3 4]
 [3 4 5]
 [4 5 6]]
```

#### 3.2.5. Copying: A precautionary Tale
Numpy arrays are mutable, and assigning them directly creates a reference, not a copy.
- Use `.copy()` to create a separate array.
Example:
```python
a = np.array([1, 2, 3])
b = a
a[0] = 99
print(b) # Output: [99 2 3]  (reference)
c = a.copy()
a[0] = 42
print(c) # Output: [99 2 3]  (independent copy)
```

#### 3.2.6. Per-Axis Operations
Many numpy functions can operate along specific axes of a multidimensional array.
- Summing along rows or columns using `axis` parameter: `axis=0` operates downwards, `axis=1` operates horizontally.
Example:
```python
array = np.array([[1, 2, 3], [4, 5, 6]])
print(np.sum(array, axis=0)) # Output: [5 7 9]  (column sum)
print(np.sum(array, axis=1)) # Output: [6 15]   (row sum)
```

#### 3.2.7. Performance
Numpy is faster than native Python for large datasets due to its C-based implementation and optimised libraries.
- Operations on numpy arrays are faster than equivalent list operations.

### 3.3. Object-oriented Programming
- Method: instance / static / class methods. <br/>
  i. Instance methods: Require `self` as the first parameter, used to interact with the instance's attributes and other instance methods. <br/>
  Example:
  ```python
  class Circle:
  	def __init__(self, radius):
  		self.radius = radius
  	def area(self):
  		return 3.14 * self.radius ** 2

  >>> circle = Circle(5)
  >>> print(circle.area())
  78.53975
  ```
  ii. Static methods: Doesn't depend on instances or class, defined with `@staticmethod` decorator, doesn't require `self` or `cls` parameter. <br/>
  Example:
  ```python
  class Circle:
  	@staticmethod
  	def pi():
  		return 3.14

  >>> print(Cicle.pi())
  3.14
  ```
  iii. Class methods: Work with the class itself, not the instance, defined with `@classmethod` decorator, takes `cls` as the first parameter. <br/>
  Example:
  ```python
  class Circle:
  	scale = 1

  	@classmethod
  	def set_scale(cls, new_scale):
  		cls.scale = new_scale

  >>> Circle.set_scale(2)
  >>> print(Circle.scale)
  2
  ```
- Use the predefined Complex class in our code by `from <filename> import Complex`
- Particular instance method `__str__()` prints instances of the class using the standard `print()` command.
  ```python
  class Complex:
  	# ...
   	def __str__(self):
   		if self.imag >= 0.0:
   			return "{0:f} + {1:f} i".format(self.real, self.imag)
   		else:
   			return "{0:f} - {1:f} i".format(self.real, -self.imag)
	...
  >>> c = Complex(1.0, -3.0)
  >>> print(c)
  1.000000 - 3.000000 i
  ```
- Magic methods: Define the behaviour of objects for specific operations, allow customisation of python's built-in behaviours, i.e., addition, string representation. <br/>
Example:
```python
class Vector:
	def __init__(self, x, y):
		self.x = x
    		self.y =y

    	def __add__(self, other):
    		return Vector(self.x + other.x, self.y + other.y)

    	def __str__(self):
    		return f"Vector({self.x}, {self.y})"

>>> v1 = Vector(1, 2)
>>> v2 = Vector(3, 4)
>>> v3 = v1 + v2
>>> print(v3)
Vector(4, 6)
```
Other magic methods: <br/>
|     Method      |     Operator     |
| --------------- | ---------------- |
|     __add__     |        +         |
|     __sub__     |        -         |
|     __mul__     |        *         |
|     __truediv__ |        /         |
|     __mod__     |        %         |
|     __pow__     |        **        |
|    __floordiv__ |        //        |

## Chapter 4. Modelling
### 4.1 The simulation Process
#### 4.1.1. Analytical and numerical solutions
Analytical solutions involve solving equations symbolically, while numerical solutions approximate solutions using computational methods. <br/>
**Example**: Solving N2 for a falling object:
- Analytical: $y(t)=y_0+v_0t-\frac{1}{2}gt^2$
- Numerical: Use Euler's method to step through $y(t)$ iteratively.

#### 4.1.2. Common numerical models in scientific simulations
- **ODEs** for time-dependent systems.
- **PDEs** for spatial and temporal problems.
- **Monte Carlo Methods** for probabilistic simulations.

#### 4.1.3.Parallel Programming
Parallel programming divides tasks across multiple processors to improve performance in computational simulations.
- Python's `multiprocessing` library parallelise a simulation of particle interactions.

### 4.2. Algorithms
#### 4.2.1. Converting a mathematical representation to an algorithm
Start with equations, break them into steps, and translate them into pseudocode or actual code.

#### 4.2.2. Supporting Algorithms
Supporting algorithms handle tasks like data preprocessing, boundary conditions, and data storage during simulations.

#### 4.2.3. Converting an algorithm to actual code
Use modular, well-documented code to ensure clarity and reusability.

#### 4.2.4. A more complicated examples
Tackling advanced simulations often involves combining multiple algorithms and testing for edge cases.

### 4.3. Representing values on a computer
#### 4.3.1. Representing integer numbers
Integers are stored exactly but have a range limitation based on the number of bits used.
- A 32-bit signed integer ranges from $-2^{31}$ to $2^{31}-1$.

#### 4.3.2. Representing real numbers
Real numbers (floating-point) are stored approximately due to finite precision.
- Python's `float` type uses IEEE 754 double precision, accurate to about 15 decimal digits

### 4.4. Errors and Debugging
- Syntax errors: Errors in the structure of code, i.e., missing colons, incorrect indentation.
- Semantic errors: The code runs but does not do what is intended.
- Logic errors: Errors in the program's logic produce incorrect results.
- Run-time errors: Errors that occur during execution, i.e., division by zero.
- Incorrect results: Errors due to numerical instability or incorrect algorithms. <br/>
  Example: <br/>
  ```python
  print(1e20 + 1 - 1e20) # returns 0.0 instead of 1. Idiot.
  ```

## Appendix C. Modern computer hardware
### C.1. Central Processing Unit (CPU)
The CPU is the brain of the computer, performing basic arithmetic, logic, control, and input/output operations.
- It connects to main memory via a bus, enabling fast data transfer.
- Tasks are performed through specialised components like FPU and IU. <br/>

|     Floating point operations                               |            Integer operations     |
|     ------------------------------------------------------- | --------------------------------- |
| Perform with limited precision, leading to rounding errors. | Perform exactly and store exactly |

- SIMD (Single Instruction, Multiple Data): Modern CPUs can perform the same operation on multiple data points simultaneously by **vector operations** on floating-point numbers.
- Multicore CPUs

### C.2. Memory
- Variables and data are stored in memory, with addresses pointing to their locations.
  ```python
  x = 42
  print(id(x)) # Returns the memory address of the variable `x`.
  ```
- Copying multiple objects like arrays requires special care. Direct assignment creates a reference, not a new copy.
- CPUs use a hierarchy of caches (L1, L2, L3) to reduce the latency of accessing frequently used data. A simulation involving repetitive calculations benefits from cache optimisation by keeping frequently accessed data closer to the CPU.

### C.3. Supercomputers
Supercomputers are high-performance systems used for large-scale simulations and computations.

### C.4. Accelerators
Accelerators like GPUs and TPUs are specialised hardware for parallel processing.
- GPUs handles matrix and vector operations, crucial for simulations and ML tasks.
- TPUs are optimised for NN computations.
```python
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor = torch.tensor([1.0, 2.0, 3.0]).to(device)
print(tensor) # runs computations on GPU if available.
```

# Introductory lecture & Workshop mini-lectures
## Introductory lecture
- Common python data types

|     float      |     real numbers, stored to ~16 sig figs     |    3.14, 2.5, 6.022e23   |
| -------------- |  ------------------------------------------- |  ----------------------  |
|     int        |     (integer) whole number value             |    1, 99, 2474           |
|     str        |     (string) text                            |    'AB', 'Hello', '2.5'  |
|     bool       |     (boolean) T or F                         |    True, False           |

- Composite types

|     List       |     L = [0, 1, 1.33, "abc", word, pi]        |    Mutable               |
| -------------- |  ------------------------------------------- |  ----------------------  |
|     Tuple      |     T = (1, 2, 3, "abc")                     |    Immutable             |
|  Dict(ionary)  |     D = {"key" : "value"}                    |                          |

### File input/output <br/>
#### i. Writing to a file: <br/>
- Use `open()` function with the mode `"w"` to open a file for writing.
- `write()` method writes text to the file.
  ```python
  with open("filename.txt", "w") as f:
  	f.write("Hello there\n")
  ```
#### ii. Reading from a file: <br/>
- Use `open()` with the mode `"r"` to open a file for reading.
- `readline()` method reads a single line from the file. <br/>
```python
with open("filename.txt", "r") as f:
	greeting = f.readline()
print(greeting) # Return: Hello there
```
#### iii. String functions: <br/>
- Functions like `.split()` can process text read from a file, i.e., splitting lines into words or values.
```python
line = "Hello there\n"
words = line.split() # splits into ['Hello', 'there']
```

### Command line
```python
python newton.py 9.81 apple
%run newton.py 9.81 apple
```
```python
import sys
print(sys.argv)
# this is a list and will print:
# ["newton.py", "9.81", "apple"]
G = float(sys.argv[1])
```

## Workshop 2 mini-lecture: Object-oriented programming
Objects group data and functions for accessing and operating on that data
- Class: A blueprint(template) for creating objects. Define the data (attribute) and behaviour (methods) of an object.
- Instance: A specific object created from a class.

Python is designed for OOP.
Structure of a Python object:
- Attributes: data values for one instance.
- Methods: functions acting on one instance.
- Special methods: built-in methods for special behaviour, i.e., `__init__`
```python
my_circle = Circle(r, x0, y0, colour)    # Making an instance
c = my_circle.radius                     # Attribute
A = my_circle.compute_area()             # Calling a method
```

in **Circle** class:
```python
import numpy as np
class Circle:
	def __init__(self, radius, x_pos, y_pos, colour):
		self.radius = radius
		self.position = np.array([x_pos, y_pos])
		self.colour = colour

	def compute_area(self):
		return np.pi * self.radius**2
```
- `class Circle` defines the name of the class, starts the methods.
- `__init__` defines how to create a new circle object. It usually set all attributes that an object has.
- `self` is the first argument to most python methods, indicating "this specific Circle".
- `self.colour = colour` is an attribute, accessed by `c = Circle(...)` -> `c.colour`.
- `compute_area` function is a (non-special) method of the class. Methods can have arguments and take another instance like `def compute_overlap(self, other_circle):`

Using **Circle** class
```python
# Import the Circle class from other file circle.py
from circle import Circle

# Initialise two circle object instances
circle1 = Circle(1.0, 2.0, 0.0, "red")
circle2 = Circle(1.0, 3.0, 3.0, "blue")

# Obtain some results using methods
area1 = circle1.compute_area()
overlap = circle1.compute_overlap(circle2)

# Modify instance properties directly
circle1.radius = 2.0
circle1.position[0] = -3.5

circle2.radius = circle1.radius
```

Brainstorming process: what algorithm? -> what classes? -> what properties of class? -> what methods/types?

## Workshop 3 mini-lecture: Numpy arrays
Useful external modules:
- numpy: core numerical tool
- scipy: uses numpy, many algorithms for mathematical tasks
- matplotlib: make plots

Arrays
```python
a = np.array([4.4, 5.5, 6.6, 7.7])       # example
x = np.zeros((100, 3))                   # 2D numpy array with 100 rows and 3 columns; initialised to 0.0; default dtype 'float'
u = np.zeros((1200, 1600, 3))            # 3D numpy array with height 1200, width 1600, depth 3
```

|              |      list      |     array      |
| ------------ | -------------- | -------------- |
| support      | built-in []    | requires numpy |
| indices      | one index (1D) | multiple       |
| element type | can mix        | all the same   |
| maths        | --             | many methods   |

```python
import numpy as np

# ------- Creating arrays -------
# makes array of ten 0.0s
x = np.zeros(10)

# creates 10x2 array, all 1
y = np.ones((10,2))

# 0.0, 1.0, 2.0, 3.0, 4.0
z = np.arange(5.)

# load a text file
u = np.genfromtxt(filename)

# ------- Accessing elements -------
# pick out one element (0=first)
a = x[5]

# modify elements
x[5] += 1
x[0] = x[1] + x[2]

# accessing higher dimensional arrays
b = y[5, 1]

# extract a whole column/row
c = y[: , 0]      # select all rows, first column (index 0)

# ------- Checking array size and shape -------
print(x.size) # 10
print(y.shape) # (10, 2)
print(y.ndim) # 2

# ------- Reshape -------
a = x.reshape(20) # must maintain total size

# ------- Maths with arrays (repeated contents are excluded) -------
make_identity = np.identity(2)
cross_product =  np.cross(a,b)
```

## Workshop 4 mini-lecture: Style
Naming
- Clear name of variables and functions, avoid ambiguity
- Methods and functions lower case verbs, with underscores between words
- Top-level constants in upper-case
- Classes in CamelCase

Mind spacing as well

Code structure: DRY

Comment Syntax: # and """Docstrings"""
- # for ppl looking at the code itself.
- Docstring for ppl using the code.
- Avoid multi-inline commenting style
