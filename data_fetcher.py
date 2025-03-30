import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class DataFetcher:
    def __init__(self):
        # Define asset symbols and their corresponding Yahoo Finance symbols
        self.asset_symbols = {
            'BITCOIN': 'BTC-USD',
            'ETHEREUM': 'ETH-USD',
            'SP500': '^GSPC',
            'NASDAQ': '^IXIC',
            'DOW': '^DJI',
            'GOLD': 'GC=F',
            'OIL': 'CL=F',
            'DXY': 'DX-Y.NYB'
        }
        
        # Define data intervals
        self.intervals = {
            'daily': '1d',
            'weekly': '1wk',
            'monthly': '1mo'
        }
    
    def fetch_data(self, asset_type, interval='daily', period='5y'):
        """
        Fetch data for the specified asset type and interval.
        
        Args:
            asset_type (str): One of the supported asset types
            interval (str): 'daily', 'weekly', or 'monthly'
            period (str): Time period to fetch (e.g., '1y', '5y', '10y')
            
        Returns:
            pandas.DataFrame: DataFrame with OHLCV data
        """
        if asset_type not in self.asset_symbols:
            raise ValueError(f"Unsupported asset type: {asset_type}")
        
        symbol = self.asset_symbols[asset_type]
        yf_interval = self.intervals.get(interval, '1d')
        
        try:
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=yf_interval)
            
            # Reset index to make Date a column
            df = df.reset_index()
            
            # Rename columns to match our analyzer's expectations
            df = df.rename(columns={
                'Date': 'Date',
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            
            return df
            
        except Exception as e:
            raise Exception(f"Error fetching data for {asset_type}: {str(e)}")
    
    def get_available_assets(self):
        """Return list of available asset types."""
        return list(self.asset_symbols.keys())
    
    def get_available_intervals(self):
        """Return list of available intervals."""
        return list(self.intervals.keys()) 