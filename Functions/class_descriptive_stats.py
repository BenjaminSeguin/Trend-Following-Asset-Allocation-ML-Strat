import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class DescriptiveStatistics:
    def __init__(self, returns: pd.DataFrame):
        """
        Initialize the DescriptiveStatistics class

        returns: A Pandas Series or DataFrame of portfolio returns
        """
        self.returns = returns


    def sharpe_ratio(self):
        excess_return = self.returns.mean() 
        return excess_return / self.returns.std()

    def max_drawdown(self):
        cumulative_returns = (1 + self.returns).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return drawdown.min()

    def turnover(self, weights_prev, weights_curr):
        return np.sum(np.abs(weights_curr - weights_prev)) / 2

    def cumulative_return(self):
        return (1 + self.returns).cumprod()[-1] - 1

    def mean_return(self):
        return self.returns.mean()

    def mean_volatility(self):
        return self.returns.std()


    def performance_summary(self, factors_df=None, weights_prev=None, weights_curr=None):
        """
        Generate a DataFrame containing all performance metrics.

        factors_df: DataFrame with Fama-French factors for alpha calculation
        weights_prev: Previous portfolio weights for turnover calculation
        weights_curr: Current portfolio weights for turnover calculation

        return: DataFrame with performance metrics
        """
        metrics = {
            "Sharpe Ratio": self.sharpe_ratio(),
            "Max Drawdown": self.max_drawdown(),
            "Cumulative Return": self.cumulative_return(),
            "Mean Return": self.mean_return() * 12, 
            "Mean Volatility": self.mean_volatility() * np.sqrt(12)
        }

  
        return pd.DataFrame(metrics, index=['Value']).T


