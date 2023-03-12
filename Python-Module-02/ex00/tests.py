import unittest
from ft_map import ft_map
from ft_reduce import ft_reduce
from ft_filter import ft_filter

class TestFunctions(unittest.TestCase):

    def test_ft_map(self):
        # Test with list
        self.assertEqual(list(ft_map(lambda x: x*2, [1,2,3])), [2,4,6])
        # Test with tuple
        self.assertEqual(list(ft_map(lambda x: x*2, (1,2,3))), [2,4,6])
        # Test with generator expression
        self.assertEqual(list(ft_map(lambda x: x*2, (x for x in [1,2,3]))), [2,4,6])
        # Test with invalid input
        self.assertIsNone(ft_map(lambda x: x*2, 123))

    def test_ft_filter(self):
        # Test with list
        self.assertEqual(list(ft_filter(lambda x: x%2==0, [1,2,3,4,5])), [2,4])
        # Test with tuple
        self.assertEqual(list(ft_filter(lambda x: x%2==0, (1,2,3,4,5))), [2,4])
        # Test with generator expression
        self.assertEqual(list(ft_filter(lambda x: x%2==0, (x for x in [1,2,3,4,5]))), [2,4])
        # Test with invalid input
        self.assertIsNone(ft_filter(lambda x: x%2==0, 123))

    def test_ft_reduce(self):
        # Test with list
        self.assertEqual(ft_reduce(lambda x,y: x+y, [1,2,3]), 6)
        # Test with tuple
        self.assertEqual(ft_reduce(lambda x,y: x+y, (1,2,3)), 6)
        # Test with generator expression
        self.assertEqual(ft_reduce(lambda x,y: x+y, (x for x in [1,2,3])), 6)
        # Test with string
        self.assertEqual(ft_reduce(lambda x,y: x+y, 'Hello World'), 'Hello World')
        # Test with invalid input
        self.assertIsNone(ft_reduce(lambda x,y: x+y, 123))
        
if __name__ == '__main__':
    unittest.main()
