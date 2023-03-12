import unittest
from TinyStatistician import TinyStatistician

class TestTinyStatistician(unittest.TestCase):
    
    def setUp(self):
        self.tstat = TinyStatistician()
        self.a = [1, 42, 300, 10, 59]
    
    def test_mean(self):
        self.assertAlmostEqual(self.tstat.mean(self.a), 82.4, places=2)
        
    def test_median(self):
        self.assertAlmostEqual(self.tstat.median(self.a), 42.0, places=2)
    
    # def test_quartiles(self):
    #     self.assertEqual(self.tstat.quartiles(self.a), [10.0, 59.0])
        
    def test_var(self):
        self.assertAlmostEqual(self.tstat.var(self.a), 12279.44, places=2)
    
    def test_std(self):
        self.assertAlmostEqual(self.tstat.std(self.a), 110.81263465868862, places=2)

if __name__ == '__main__':
    unittest.main()
