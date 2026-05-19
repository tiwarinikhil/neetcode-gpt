import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        n = len(X)
        m = len(X[0])
        res = []
        for i in range(n):
            curr_res = 0
            for j in range(m):
                curr_res = curr_res + X[i][j] * weights[j]
                # print(curr_res)
            res.append(curr_res)
        return np.round(res,5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        n = len(model_prediction)
        m = len(model_prediction[0])

        error = 0
        for i in range(n):
            curr_error = 0
            for j in range(m):
                curr_error = curr_error + (model_prediction[i][j]-ground_truth[i][j])**2
            error = error + curr_error
        error = error/n
        return round(error,5) 
