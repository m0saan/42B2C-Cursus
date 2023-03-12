import numpy as np
import math

class TinyStatistician:
    
    def mean(self, x):
        if not len(x): return None
        return float(np.sum(x) / len(x))
    
    def median(self, x):
        if not len(x): return None
        
        x = sorted(x)
        n = len(x)
        if len(x) % 2 == 0: return float((x[n // 2 - 1] + x[n//2]) / 2)
        return float(x[n//2])
    
    def var(self, x):
        if not len(x): return None
        
        mean = self.mean(x)
        x = np.array(x)
        return np.sum((x - mean) ** 2) / len(x)
    
    def std(self, x):
        return math.sqrt(self.var(x))
        
        
        
tstat = TinyStatistician()
a = [1, 42, 300, 10, 59]
print(tstat.mean(a))
# Expected result: 82.4
print(tstat.median(a))
# Expected result: 42.0
# print(tstat.quartile(a))
# Expected result: [10.0, 59.0]
print(tstat.var(a))
# Expected result: 12279.439999999999
print(tstat.std(a))
# Expected result: 110.81263465868862