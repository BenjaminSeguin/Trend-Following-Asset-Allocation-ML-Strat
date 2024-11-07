import numpy as np
import pandas as pd
from typing import List
from abc import ABC, abstractmethod

# Abstract base class to define a common interface for all signal types
class Signal(ABC):
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the Signal class with data
        
        Parameters:
        ----------
        data: pd.DataFrame
            The dataframe containing asset prices
        """
        self.data = data
        # Calculate returns and align with data
        self.returns = self.data.pct_change()
        # Ensure both dataframes have the same index
        self.data, self.returns = self.data.align(self.returns, axis=0)
        

    def calculate_returns(self, days, asset):
        """
        Compites returns over a certain horizon for a specific asset
        
        Parameters:
        ----------
        days: int
            The horizon over which we wish to compute returns
        asset: str
            The asset for which we wish to compute the returns

        Returns:
        -------
        pd.Series
            A series representing the return of the asset
        """

        return self.data[asset].pct_change(days).dropna()

    def moving_average(self, days, asset):
        """
        Calculate the moving average for a specified number of days for a given asset
        
        Parameters:
        ----------
        days: int
            The window size for the moving average calculation
        asset: str
            The name of the asset for which we wish to compute moving average
        
        Returns:
        -------
        pd.Series
            A series representing the moving average of the asset
        """
        return self.returns[asset].rolling(window=days).mean().dropna()
        

    def volatility(self, days, asset):
        """
        Calculate the rolling standard deviation for a specified number of days for a given asset
        
        Parameters:
        ----------
        days: int
            The window size for the volatility calculation
        asset: str
            The name of the asset
        
        Returns:
        -------
        pd.Series
            A series representing the rolling standard deviation of the asset
        """
        return self.returns[asset].rolling(window=days).std()
    
    def ema(self, days, asset):
        """
        Calculate the exponential moving average for a specified number of days for a given asset
        
        Parameters:
        ----------
        days: int
            The window size for the exponential moving average calculation
        asset: str
            The name of the asset
        
        Returns:
        -------
        pd.Series
            A series representing the exponential moving average of the asset
        """
        return self.returns[asset].ewm(span=days).mean().shift().dropna()

    @abstractmethod
    def calculate(self):
        """
        Abstract method to calculate the signal
        Must be implemented by subclasses
        """
        pass


class Return(Signal):
    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
        self.signal = None
        self.data = self.data.groupby(self.data.index).first().sort_index()

    def condition(self, days, asset):
        """
        Calculate the signal value based on historical volatility

        Parameters:
        ----------
        days: int
            The window size used for rolling standard deviation and return
        asset: str
            The asset column used in calculations
        
        Returns:
        -------
        pd.Series
            A series containing the calculated signal values
        """
        returns = self.calculate_returns(days, asset)
        volatility = self.volatility(days, asset) * np.sqrt(days)
        
        # Ensure returns and volatility have the same index
        common_index = returns.index.intersection(volatility.index)
        returns = returns.loc[common_index]
        volatility = volatility.loc[common_index]
        
        # Calculate signal values based on past volatility
        signal_values = returns #/ volatility
        
        return signal_values

    def Return_signal(self, days, asset):
        """
        Generate return-based signals using the calculated signal values.
        
        Parameters:
        ----------
        days: int
            The window size for calculating signal values.
        asset: str
            The asset used for generating the signals.
        
        Returns:
        -------
        pd.DataFrame
            A dataframe containing the calculated signal values.
        """
        signal_df = pd.DataFrame(index=self.returns.index)
        signal_values = self.condition(days, asset)

        # Assign the signal values directly to the dataframe
        signal_df[f'Signal Return {asset} {days}D'] = signal_values

        # Before returning, ensure no duplicate index values
        return_df = signal_df.groupby(signal_df.index).first().sort_index()
        return return_df

    def calculate(self, days_list, asset):
        """ 
        Calculate return signals for a single asset and multiple time windows
        
        Parameters:
        ----------
        days_list: List[int]
            List of window sizes for moving average calculations
        asset: str
            The single asset for which signals will be calculated
        
        Returns:
        -------
        pd.DataFrame
            A dataframe containing all the generated signals for the asset
        """
        self.signal = pd.DataFrame(index=self.returns.index)

        # Loop over days to generate return signals for the asset
        for days in days_list:
            signal_df = self.Return_signal(days, asset)
            self.signal = pd.concat([self.signal, signal_df], axis=1)

        return self.signal


class MovingAverage(Signal):
    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
        self.signal = None

    def condition(self, days, asset):
        """
        Calculate the long and short signal values based on the exponential moving average 
        of returns and volatility

        Parameters:
        ----------
        days : int
            The window size for calculating the moving average and volatility
        asset : str
            The asset used for generating the signals

        Returns:
        -------
        pd.Series
            A series containing the calculated signal values based on the EMA of returns.
        """
        returns = self.calculate_returns(22, asset) # monthly return
        volatility = self.volatility(22, asset) # monthly volatility
        
        # Ensure returns and volatility have the same index
        common_index = returns.index.intersection(volatility.index)
        returns = returns.loc[common_index]
        volatility = volatility.loc[common_index]
        values = returns 

        signal_values = values.ewm(span=days).mean()
        
        return signal_values


    def moving_average_signal(self, days, asset):
        """
        Generate return-based signals using the calculated signal values
        
        Parameters:
        ----------
        days: int
            The window size for calculating signal values
        asset: str
            The asset used for generating the signals
        
        Returns:
        -------
        pd.DataFrame
            A dataframe containing the calculated signal values
        """
        signal_df = pd.DataFrame(index=self.returns.index)
        signal_values = self.condition(days, asset)

        # Assign the signal values directly to the dataframe
        signal_df[f'Signal EMA {asset} {days}D'] = signal_values

        return signal_df

    def calculate(self, days_list, asset):
        """ 
        Calculate moving average signals for a single asset and multiple time windows
        
        Parameters:
        ----------
        days_list: List[int]
            List of window sizes for moving average calculations
        asset: str
            The single asset for which signals will be calculated
        
        Returns:
        -------
        pd.DataFrame
            A dataframe containing all the generated signals for the asset
        """
        self.signal = pd.DataFrame(index=self.data.index)

        for days in days_list:
            signal_df = self.moving_average_signal(days, asset)
            self.signal = pd.concat([self.signal, signal_df], axis=1)

        return self.signal

class CrossMovingAverage(Signal):
    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
        self.signal = None

    def condition(self, short_days, long_days, asset):
        """
        Calculate the condition for cross-moving average signal using EMA

        Parameters:
        ----------
        short_days: int
            The window size for the short-term EMA
        long_days: int
            The window size for the long-term EMA
        asset: str
            The asset used for generating the signals

        Returns:
        -------
        pd.Series
            A series containing the value of short - long EMA
        """

        returns = self.returns[asset]
        values = returns

        short_ema = values.ewm(span=short_days, adjust=False).mean()
        long_ema = values.ewm(span=long_days, adjust=False).mean()

        # Generate signals based on EMA crossover
        signal = pd.Series(0, index=self.data.index)
        signal = short_ema - long_ema

        return signal   


    def cross_moving_average_signal(self, short_days, long_days, asset):
        """
        Generate cross-moving average signals using short-term and long-term moving averages
        
        Parameters:
        ----------
        short_days: int
            The window size for the short-term moving average
        long_days: int
            The window size for the long-term moving average
        asset: str
            The asset used for generating the signals
        
        Returns:
        -------
        pd.DataFrame
            A dataframe containing the value of short - long EMA for a specific asset and rightly indexed
        """
        signal_df = pd.DataFrame(index=self.returns.index)
        signal_values = self.condition(short_days, long_days, asset)

        # Assign the signal values directly to the dataframe
        signal_df[f'Signal MACD {asset} {short_days}D {long_days}D'] = signal_values

        return signal_df

    def calculate(self, short_long_days_list: List[tuple], asset):
        """
        Calculate cross-moving average signals for multiple assets and time windows
        
        Parameters:
        ----------
        short_long_days_list: List[tuple]
            List of short-term and long-term moving average window sizes
        asset: str
            The single asset for which signals will be calculated
        
        Returns:
        -------
        pd.DataFrame
            A dataframe containing all the generated signals for specific asset and moving average combination
        """
        self.signal = pd.DataFrame(index=self.data.index)

        # Loop over short-long day combinations to generate cross-moving average signals

        for short_days, long_days in short_long_days_list:
            signal_df = self.cross_moving_average_signal(short_days, long_days, asset)
            self.signal = pd.concat([self.signal, signal_df], axis=1)

        return self.signal


class BollingerBands(Signal):
    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
        self.signal = None

    def bollinger_signal(self, days, asset):
        """
        Generate Bollinger Bands signals based on moving average and volatility
        
        Parameters:
        ----------
        days: int
            The window size for moving average and volatility calculations
        asset: str
            The asset used for generating the signals
        
        Returns:
        -------
        pd.DataFrame
            A dataframe containing the signals (1 for long, -1 for short, 0 for hold)
        """
        signal_df = pd.DataFrame(index=self.data.index)
        
        # Calculate moving average and standard deviation
        ma = self.returns[asset].rolling(window=days).mean()
        std = self.returns[asset].rolling(window=days).std()
        
        # Calculate upper and lower Bollinger Bands
        upper_band = ma + 1 * std
        lower_band = ma - 1 * std
        
        # Generate signals
        signals = pd.Series(0, index=self.data.index)
        signals[self.returns[asset] > upper_band] = 1  # Long signal
        signals[self.returns[asset] < lower_band] = -1  # Short signal
        
        signal_df[f'Signal Bollinger {asset} {days}D'] = signals
        
        return signal_df

    def calculate(self, days_list: List[int], asset):
        """
        Calculate Bollinger Bands signals per asset and time windows
        
        Parameters:
        ----------
        days_list: List[int]
            List of window sizes for moving average and volatility calculations
        asset: str
            The single asset for which signal will be calculated
        
        Returns:
        -------
        pd.DataFrame
            A dataframe containing all the generated signals per asset and time window
        """
        self.signal = pd.DataFrame(index=self.data.index)

        # Loop over days to generate Bollinger Bands signals

        for days in days_list:
            signal_df = self.bollinger_signal(days, asset)
            self.signal = pd.concat([self.signal, signal_df], axis=1)

        return self.signal

class MinMaxBreakout(Signal):
    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
        self.signal = None

    def min_max_breakout_signal(self, window, asset):
        """
        Generate signals based on Min/Max breakout
        
        Parameters:
        ----------
        window: int
            The number of days to look back for the min/max price
        asset: str
            The asset used for generating the signals
        
        Returns:
        -------
        pd.DataFrame
            A dataframe containing the signals (1 for long, -1 for short, 0 for hold)
        """
        signal_df = pd.DataFrame(index=self.data.index)
        
        # Calculate rolling max and min
        rolling_max = self.data[asset].rolling(window=window).max().shift(1)
        rolling_min = self.data[asset].rolling(window=window).min().shift(1)
        
        # Generate signals
        signals = pd.Series(0, index=self.data.index)
        signals[self.data[asset] > rolling_max] = 1  # Long signal
        signals[self.data[asset] < rolling_min] = -1  # Short signal
        
        signal_df[f'Signal Breakout {asset} {window}D'] = signals
        
        return signal_df

    def calculate(self, window_list: List[int], asset):
        """
        Calculate breakout signals per asset and window sizes
        
        Parameters:
        ----------
        window_list: List[int]
            A list of window sizes for the min/max breakout calculations
        asset: str
            The single asset for which signals will be calculated
        
        Returns:
        -------
        pd.DataFrame
            A dataframe containing all the generated signals per asset and window
        """
        self.signal = pd.DataFrame(index=self.data.index)

        # Loop over window sizes to generate signals

        for window in window_list:
            signal_df = self.min_max_breakout_signal(window, asset)
            self.signal = pd.concat([self.signal, signal_df], axis=1)

        return self.signal

