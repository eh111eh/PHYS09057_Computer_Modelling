- Role of `__init__` and `self`:
```python
 class A(object):
		def __init__(self):
			self.x = 'Hello'
		def method_a(self, foo):
			print self.x + ' ' + foo
	The self variable represents the instance of the object itself.
	The __init__() method takes arguments and assigns them to properties of the object.
```
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
- Numpy

- Object-oriented Programming
	- Use the predefined Complex class in our code by `from <filename> import Complex`