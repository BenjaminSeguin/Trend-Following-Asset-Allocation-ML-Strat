import pandas as pd
import numpy as np
from functools import reduce 
from typing import List

class DataReader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = None

    def load_data(self, sheet_names: List[str]) -> pd.DataFrame:
        """Load data from multiple Excel sheets and merge them into a single DataFrame."""
        sheet_data = []
        for name  in sheet_names:
            try:
                df = pd.read_excel(self.filepath, sheet_name=name)
                sheet_data.append(df)
            except ValueError as e:
                print(f"Error loading sheet {name}: {e}")
                continue

        if sheet_data:
            self.data = reduce(lambda left, right: pd.merge(left, right, on='Date', how='outer'), sheet_data)
            self.data.set_index('Date', inplace=True)
            
            # After loading the data, remove any duplicate index values
            self.data = self.data.groupby(self.data.index).first()
            self.data.sort_index(inplace=True)
        else:
            raise ValueError("No data loaded. Ensure sheet names are correct and the file exists.")
        
        return self.data
    

class DataProcessor(DataReader):
    def __init__(self, filepath: str):
        super().__init__(filepath)
    
    def process_data(self):
        pass


class SignalProcessor:
    def __init__(self, data: pd.DataFrame):
        self.data = data.groupby(data.index).first().sort_index()

    def process_signals(self, asset_name: str, signal_classes: List[object], signal_params: dict) -> pd.DataFrame:
        """Process multiple signal classes for a given asset and return a DataFrame with concatenated signals."""
        
        if asset_name not in self.data.columns:
            raise ValueError(f"Asset {asset_name} not found in the data.")
        
        signals_df = pd.DataFrame(index=self.data.index)
        
        for signal_class in signal_classes:
            signal_instance = signal_class(self.data[[asset_name]])  
            
            params = signal_params.get(signal_class.__name__, ())
      
            try:
                signal_result_df = signal_instance.calculate(*params)
                signals_df = pd.concat([signals_df, signal_result_df], axis=1)
            except ValueError as e:
                print(f"Error processing {signal_class.__name__} for {asset_name}: {str(e)}")
                continue
            
        return signals_df

    def return_target(self, asset, signal_data):
        returns = self.data[asset].pct_change()
        
        data = pd.DataFrame(index=signal_data.index)

        data[f'{asset}_Return'] = returns.reindex(signal_data.index)
        
        for column in signal_data.columns:
            data[column] = signal_data[column]
        
        return data
