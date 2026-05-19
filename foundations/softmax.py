import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        res = z.copy()
        n = len(z)
        demoninator = 0
        maxi = max(z)
        for i in range(n):
            demoninator = demoninator+ np.exp(z[i]- maxi)

        for i in range(n):
            res[i] = (np.exp(z[i]-maxi)) / demoninator
        
        return np.round(res, 4)
