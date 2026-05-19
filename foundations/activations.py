import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        n = len(z)
        res = np.zeros(n)
        for i in range(n):
            res[i]= 1/( 1+ np.exp(-z[i]) )
        
        return np.round(res , 5)
        


    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        n = len(z)
        res = z.copy()
        for i in range(n):
            if res[i]<=0:
                res[i]=0
        
        return res