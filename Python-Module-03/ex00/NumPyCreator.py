import numpy as np

class NumPyCreator:
    
    def from_list(self, lst, dtype=None):
        """
        Takes a list or nested lists and returns its corresponding Numpy array.

        Parameters:
        @lst (list): The list or nested list to be converted to a Numpy array.
        @dtype (optional): The data type of the resulting Numpy array. Default is None.

        Returns:
        numpy.ndarray: The Numpy array containing the elements from the list.
        """
        if dtype is None:
            return np.array(lst)
        else:
            return np.array(lst, dtype=dtype)
    
    def from_tuple(self, tpl, dtype=None):
        """
        Takes a tuple or nested tuples and returns its corresponding Numpy array.

        Parameters:
        @tpl (tuple): The tuple or nested tuple to be converted to a Numpy array.
        @dtype (optional): The data type of the resulting Numpy array. Default is None.

        Returns:
        numpy.ndarray: The Numpy array containing the elements from the tuple.
        """
        if dtype is None:
            return np.array(tpl)
        else:
            return np.array(tpl, dtype=dtype)
    
    def from_iterable(self, itr, dtype=None):
        """
        Takes a tuple or nested tuples and returns its corresponding Numpy array.

        Parameters:
        @tpl (tuple): The tuple or nested tuple to be converted to a Numpy array.
        @dtype (optional): The data type of the resulting Numpy array. Default is None.

        Returns:
        numpy.ndarray: The Numpy array containing the elements from the tuple.
        """
        if dtype is None:
            return np.fromiter(itr, dtype=np.float64)
        else:
            return np.fromiter(itr, dtype=dtype)
    
    def from_shape(self, shape, value=0, dtype=None):
        """
        Returns an array filled with the same value.
        The first argument is a tuple which specifies the shape of the array,
        and the second argument specifies the value of the elements. This value must be 0 by default.

        Parameters:
        @shape (tuple): A tuple specifying the shape of the Numpy array.
        @value (optional): The value of the elements in the Numpy array. Default is 0.
        @dtype (optional): The data type of the resulting Numpy array. Default is None.

        Returns:
        numpy.ndarray: The Numpy array of the given shape, filled with the specified value.
        """
        if dtype is None:
            return np.full(shape, value)
        else:
            return np.full(shape, value, dtype=dtype)
    
    def random(self, shape, dtype=None):
        """
        Returns an array filled with random values. It takes as an argument a tuple which specifies the shape of the array.

        Parameters:
        @shape (tuple): A tuple specifying the shape of the Numpy array.
        @dtype (optional): The data type of the resulting Numpy array. Default is None.

        Returns:
        numpy.ndarray: The Numpy array of the given shape, filled with random values.
        """
        if dtype is None:
            return np.random.rand(*shape)
        else:
            return np.random.rand(*shape).astype(dtype)
    
    def identity(self, n, dtype=None):
        """
        Returns an array representing the identity matrix of size n.

        Parameters:
        @n (int): The size of the identity matrix.
        @dtype (optional): The data type of the resulting Numpy array. Default is None.

        Returns:
        numpy.ndarray: The Numpy array representing the identity matrix of size n.
        """
        if dtype is None:
            return np.eye(n)
        else:
            return np.eye(n, dtype=dtype)
