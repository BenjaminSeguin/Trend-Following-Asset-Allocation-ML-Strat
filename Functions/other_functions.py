import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt


class PortfolioInputs():
    def __init__(self, data: pd.DataFrame):
        self.data = data.pct_change()

     
    def calculate_covariance_matrix(self) -> pd.DataFrame:
        """
        Calculate the covariance matrix over the specified rolling window.

        Parameters:
        ----------
        window: int
            The window size for calculating the covariance matrix.

        Returns:
        -------
        cov_matrix: pd.DataFrame
            The covariance matrix of asset returns.
        """

        self.data = self.data.dropna()

        cov_matrix = self.data.cov() * 252    

        return cov_matrix
    
    def calculate_expected_returns(self):

        return self.data.mean() * 252
    

def remove_words_at_positions(col_name, positions):
    words = col_name.split()
    words_filtered = [word for idx, word in enumerate(words) if idx not in positions]
    new_col_name = ' '.join(words_filtered)
    return new_col_name