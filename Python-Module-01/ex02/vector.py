class Vector:
    """
    A class representing a mathematical vector.

    Attributes:
        values (list of list of floats): A list of lists of floats that represents the vector.
        shape (tuple of 2 integers): A tuple of two integers that represents the shape of the vector.
            The first integer is the number of rows and the second integer is the number of columns.

    Methods:
        __init__(self, values): Constructs a new vector object with the given values.
        __str__(self): Returns a string representation of the vector.
        __add__(self, other): Adds another vector to this vector and returns the result as a new vector.
        __sub__(self, other): Subtracts another vector from this vector and returns the result as a new vector.
        __mul__(self, other): Multiplies this vector by a scalar or another vector and returns the result as a new vector.
        dot_product(self, other): Computes the dot product of this vector and another vector.
    """

    def __init__(self, values):
        """
        Creates a Vector object.

        Args:
        - values: list of list of floats (for row vector) or list of lists of single float (for column vector).

        Attributes:
        - values: list of list of floats (for row vector) or list of lists of single float (for column vector).
        - shape: tuple of 2 integers: (1,n) for a row vector of dimension n or (n,1) for a column vector of dimension n.
        """
    
        self.values = values
        self.shape = self._get_shape()

    def _get_shape(self):
        if len(self.values) == 1:
            return (1, len(self.values[0]))
        elif len(self.values[0]) == 1:
            return (len(self.values), 1)
        else:
            raise ValueError("Invalid vector shape")
        
    def _is_valid_shapes(self, other):
        return self.shape == other.shape
    
    def __eq__(self, other):
        if isinstance(other, Vector):
            return self.values == other.values
        return False

    def __str__(self):
        """
        Returns a string representation of the Vector object.

        Returns:
        - String representation of the Vector object.
        """
        return f"Vector({self.values})"
    
    def __repr__(self):
        """
        Returns a string representation of the Vector object.

        Returns:
        - String representation of the Vector object.
        """
        return f"Vector({self.values})"
    
    
    def __add__(self, other):
        """
        Adds two vectors of the same shape element-wise.

        Args:
        - other: Vector object of the same shape.

        Returns:
        - Vector object representing the sum of the two vectors.
        """
        if not self._is_valid_shapes(other):
            raise ValueError(f"shapes does not match {self.shape} {other.shape}")
        n,m = self.shape
        result = []
        for i in range(n):
            row = []
            for j in range(m):
                row.append(self.values[i][j] + other.values[i][j])
            result.append(row)
        return Vector(result)
            
        
    
    def __radd__(self, other):
        """
        Adds two vectors of the same shape element-wise.

        Args:
        - other: Vector object of the same shape.

        Returns:
        - Vector object representing the sum of the two vectors.
        """
        return self.__add__(other)
    
    def __sub__(self, other):
        """
        Subtracts two vectors of the same shape element-wise.

        Args:
        - other: Vector object of the same shape.

        Returns:
        - Vector object representing the difference between the two vectors.
        """
        if not self._is_valid_shapes(other):
            raise ValueError("Vectors must have the same shape for subtraction")
        result = []
        for i in range(len(self.values)):
            row = []
            for j in range(len(self.values[0])):
                row.append(self.values[i][j] - other.values[i][j])
            result.append(row)
        return Vector(result)
    
    def __rsub__(self, other):
        """
        Subtracts two vectors of the same shape element-wise.

        Args:
        - other: Vector object of the same shape.

        Returns:
        - Vector object representing the difference between the two vectors.
        """
        if isinstance(other, Vector):
            if self.shape != other.shape:
                raise ValueError("Cannot subtract vectors of different shapes")
            return Vector([[other.values[i][j] - self.values[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])])
        elif isinstance(other, (int, float)):
            return Vector([[other - self.values[i][0] for i in range(self.shape[0])]])
        else:
            raise TypeError(f"unsupported operand type(s) for -: '{type(other).__name__}' and 'Vector'")

    
    def __truediv__(self, scalar):
        """
        Divides a vector by a scalar.

        Args:
        - scalar: scalar.

        Returns:
        - Vector object representing the result of the division.
        """
        
        if isinstance(scalar, (int, float)):
            if scalar == 0:
                raise ZeroDivisionError("division by zero")
            new_values = [[elem/scalar for elem in row] for row in self.values]
            return Vector(new_values)
        else:
            raise TypeError("unsupported operand type(s) for /: 'Vector' and '{}'".format(type(scalar).__name__))
        
        
    
    def __rtruediv__(self, other):
        """
        Raises a NotImplementedError with the message "Division of a scalar by a Vector is not defined here."

        Args:
        - other: scalar.

        Raises:
        - NotImplementedError with the message "Division of a scalar by a Vector is not defined here."
        """
        raise NotImplementedError("Division of a scalar by a Vector is not defined here.")



    
    def __mul__(self, other):
        """
        Multiplies a vector by a scalar.

        Args:
        - other: scalar.

        Returns:
        - Vector object representing the result of the multiplication.
        """
        n,m = self.shape
        result = []
        for i in range(n):
            row = []
            for j in range(m):
                row.append(self.values[i][j] * other)
            result.append(row)
        return Vector(result)
            
                
    
    def __rmul__(self, other):
        """
        Multiplies a vector by a scalar.

        Args:
        - other: scalar.

        Returns:
        - Vector object representing the result of the multiplication.
        """
        return self.__mul__(other)
    
    def dot(self, other):
        
        if not self._is_valid_shapes(other):
            raise ValueError(f"shapes does not match {self.shape} {other.shape}")
        
        n,m = self.shape
        result = 0.0
        for i in range(n):
            for j in range(m):
                result +=  self.values[i][j] * other.values[i][j]
        return result
    
    def T(self):
        n, m = self.shape
        result = []
        if n == 1:
            for i in range(m):
                result.append([self.values[0][i]])
            return Vector(result)
        else:
            for i in range(n):
                result.append(self.values[i][0])
            return Vector([result])
