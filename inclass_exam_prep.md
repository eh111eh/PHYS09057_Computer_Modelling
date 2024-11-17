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
#### 3.2.1 Loading numpy and creating arrays
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

#### 3.2.2 Maths with Arrays
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

#### 3.2.3 Reshaping and Slicing
Arrays can be reshaped or sliced to access specific elements or change their structure.
- Use `.reshape()` to change the shape of an array.
- Use slicing with colons(`:`) to extract parts of an array.
Example:
```python
array = np.arange(12).reshape(3, 4) # reshape into 3 rows and 4 columns
>>> print(array[:, 1])
[1, 5, 9] # Second column
```

#### 3.2.4 Broadcasting
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

#### 3.2.5 Copying: A precautionary Tale
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

#### 3.2.6 Per-Axis Operations
Many numpy functions can operate along specific axes of a multidimensional array.
- Summing along rows or columns using `axis` parameter: `axis=0` operates downwards, `axis=1` operates horizontally.
Example:
```python
array = np.array([[1, 2, 3], [4, 5, 6]])
print(np.sum(array, axis=0)) # Output: [5 7 9]  (column sum)
print(np.sum(array, axis=1)) # Output: [6 15]   (row sum)
```

#### 3.2.7 Performance
Numpy is faster than native Python for large datasets due to its C-based implementation and optimised libraries.
- Operations on numpy arrays are faster than equivalent list operations.

### 3.3 Object-oriented Programming
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
    
