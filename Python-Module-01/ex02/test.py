import unittest
import pytest
from vector import Vector

class TestVector(unittest.TestCase):

    def test_scalar_multiplication_column_vector(self):
        v1 = Vector([[0.0], [1.0], [2.0], [3.0]])
        v2 = v1 * 5
        expected = Vector([[0.0], [5.0], [10.0], [15.0]])
        self.assertEqual(v2, expected)

    def test_scalar_multiplication_row_vector(self):
        v1 = Vector([[0.0, 1.0, 2.0, 3.0]])
        v2 = v1 * 5
        expected = Vector([[0.0, 5.0, 10.0, 15.0]])
        self.assertEqual(v2, expected)

    def test_scalar_division_row_vector(self):
        v1 = Vector([[0.0, 1.0, 2.0, 3.0]])
        v2 = v1 / 2.0
        expected = Vector([[0.0, 0.5, 1.0, 1.5]])
        self.assertEqual(v2, expected)

    def test_zero_division_error(self):
        v1 = Vector([[0.0], [1.0], [2.0], [3.0]])
        with self.assertRaises(ZeroDivisionError):
            v1 / 0.0

    def test_scalar_division_not_implemented(self):
        v1 = Vector([[0.0, 1.0, 2.0, 3.0]])
        with self.assertRaises(NotImplementedError):
            2.0 / v1

    def test_column_vector_shape(self):
        v = Vector([[0.0], [1.0], [2.0], [3.0]])
        self.assertEqual(v.shape, (4, 1))

    def test_column_vector_values(self):
        v = Vector([[0.0], [1.0], [2.0], [3.0]])
        expected = [[0.0], [1.0], [2.0], [3.0]]
        self.assertEqual(v.values, expected)

    def test_row_vector_shape(self):
        v = Vector([[0.0, 1.0, 2.0, 3.0]])
        self.assertEqual(v.shape, (1, 4))

    def test_row_vector_values(self):
        v = Vector([[0.0, 1.0, 2.0, 3.0]])
        expected = [[0.0, 1.0, 2.0, 3.0]]
        self.assertEqual(v.values, expected)

    def test_transpose_column_vector(self):
        v1 = Vector([[0.0], [1.0], [2.0], [3.0]])
        v2 = v1.T()
        expected = Vector([[0.0, 1.0, 2.0, 3.0]])
        self.assertEqual(v2, expected)
        self.assertEqual(v2.shape, (1, 4))

    def test_transpose_row_vector(self):
        v1 = Vector([[0.0, 1.0, 2.0, 3.0]])
        v2 = v1.T()
        expected = Vector([[0.0], [1.0], [2.0], [3.0]])
        self.assertEqual(v2, expected)
        self.assertEqual(v2.shape, (4, 1))

    def test_dot_product(self):
        # Test dot product between two column vectors
        v1 = Vector([[0.0], [1.0], [2.0], [3.0]])
        v2 = Vector([[2.0], [1.5], [2.25], [4.0]])
        assert v1.dot(v2) == 18.0
    
        # Test dot product between two row vectors
        v3 = Vector([[1.0, 3.0]])
        v4 = Vector([[2.0, 4.0]])
        assert v3.dot(v4) == 14.0
    
        # Test dot product between a row vector and a column vector
        with pytest.raises(ValueError):
            v5 = Vector([[0.0, 1.0, 2.0, 3.0]])
            v5.dot(v1) == 14.0
            v1.dot(v5) == 14.0
        
        # Test dot product with a vector of a different shape
        v6 = Vector([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError):
            v1.dot(v6)

if __name__ == '__main__':
    unittest.main()
    