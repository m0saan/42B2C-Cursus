import unittest
import numpy as np
from numpy.testing import assert_array_equal

from NumPyCreator import NumPyCreator


class TestNumPyCreator(unittest.TestCase):
    def setUp(self):
        self.creator = NumPyCreator()
    
    def test_from_list(self):
        # Test with a simple list
        lst = [1, 2, 3]
        expected = np.array([1, 2, 3])
        result = self.creator.from_list(lst)
        assert_array_equal(result, expected)
        
        # Test with a nested list
        lst = [[1, 2], [3, 4]]
        expected = np.array([[1, 2], [3, 4]])
        result = self.creator.from_list(lst)
        assert_array_equal(result, expected)
        
        # Test with a list of tuples
        lst = [(1, 2), (3, 4)]
        expected = np.array([(1, 2), (3, 4)])
        result = self.creator.from_list(lst)
        assert_array_equal(result, expected)
        
        # Test with a dtype argument
        lst = [1, 2, 3]
        expected = np.array([1, 2, 3], dtype=np.float32)
        result = self.creator.from_list(lst, dtype=np.float32)
        assert_array_equal(result, expected)
    
    def test_from_tuple(self):
        # Test with a simple tuple
        tpl = (1, 2, 3)
        expected = np.array([1, 2, 3])
        result = self.creator.from_tuple(tpl)
        assert_array_equal(result, expected)
        
        # Test with a nested tuple
        tpl = ((1, 2), (3, 4))
        expected = np.array([[1, 2], [3, 4]])
        result = self.creator.from_tuple(tpl)
        assert_array_equal(result, expected)
        
        # Test with a dtype argument
        tpl = (1, 2, 3)
        expected = np.array([1, 2, 3], dtype=np.float32)
        result = self.creator.from_tuple(tpl, dtype=np.float32)
        assert_array_equal(result, expected)
    
    def test_from_iterable(self):
        # Test with a simple iterable
        itr = [1, 2, 3]
        expected = np.array([1, 2, 3])
        result = self.creator.from_iterable(itr)
        assert_array_equal(result, expected)
        
        # Test with a range object
        itr = range(1, 4)
        expected = np.array([1, 2, 3])
        result = self.creator.from_iterable(itr)
        assert_array_equal(result, expected)
        
        # Test with a dtype argument
        itr = [1, 2, 3]
        expected = np.array([1, 2, 3], dtype=np.float32)
        result = self.creator.from_iterable(itr, dtype=np.float32)
        assert_array_equal(result, expected)
    
    def test_from_shape(self):
        # Test with default value
        shape = (2, 3)
        expected = np.array([[0, 0, 0], [0, 0, 0]])
        result = self.creator.from_shape(shape)
        assert_array_equal(result, expected)

        # Test with custom value
        shape = (2, 3)
        expected = np.array([[1, 1, 1], [1, 1, 1]])
        result = self.creator.from_shape(shape, value=1)
        assert_array_equal(result, expected)

        # Test with custom dtype
        shape = (2, 3)
        expected = np.array([[1, 1, 1], [1, 1, 1]], dtype=np.float32)
        result = self.creator.from_shape(shape, value=1, dtype=np.float32)
        assert_array_equal(result, expected)


    def test_random(self):
        # Test with default dtype
        shape = (2, 3)
        result = self.creator.random(shape)
        self.assertEqual(result.shape, shape)
        self.assertTrue(np.all(result >= 0) and np.all(result <= 1))
        
        # Test with custom dtype
        shape = (2, 3)
        dtype = np.int32
        result = self.creator.random(shape, dtype=dtype)
        self.assertEqual(result.shape, shape)
        self.assertEqual(result.dtype, dtype)
        
    def test_identity(self):
        # Test with default dtype
        n = 3
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        result = self.creator.identity(n)
        assert_array_equal(result, expected)
        
        # Test with custom dtype
        n = 3
        dtype = np.float64
        expected = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)
        result = self.creator.identity(n, dtype=dtype)
        assert_array_equal(result, expected)

if __name__ == '__main__':
    unittest.main()