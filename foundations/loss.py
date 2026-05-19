import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        
        n = len(y_true)
        # y_pred = [max(1e-7, min(1 - 1e-7, x)) for x in y_pred] 
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)    
        loss = 0     
        for i in range(n):
            loss = loss + y_true[i]*np.log(y_pred[i]) + (1-y_true[i]) * np.log(1-y_pred[i])
        
        loss = -(1/n)*loss
        return round(loss,4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        n = len(y_true)
        m = len(y_true[0])
        
        # y_pred = [max(1e-7, min(1 - 1e-7, y_pred[i][j])) for i in y_pred]     
        loss = 0     
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        for i in range(n):
            for j in range(m):
                y_pred[i][j] = max(1e-7, min(1 - 1e-7, y_pred[i][j]))
                loss = loss + y_true[i][j]*np.log(y_pred[i][j])

        loss = -(1/n)*loss
        return round(loss,4)
        
        
