import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cycledetector import detect_swing_lows, calculate_sma

class BressartCycleIndicator:
    def __init__(self, data, asset_type, interval='daily', custom_params=None):
        """
        Initialize the cycle indicator with price data and asset type.
        data should be a pandas DataFrame with 'Date' and 'Close' columns
        asset_type should be one of: 'BITCOIN', 'ETHEREUM', 'SP500', 'NASDAQ', 'DOW', 'GOLD', 'OIL', 'DXY'
        interval should be one of: 'daily', 'weekly'
        custom_params: optional dictionary with custom cycle parameters for non-standard assets
        """
        self.data = data.copy()
        self.interval = interval  # Store the interval
        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data.set_index('Date', inplace=True)
        elif not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)
        
        # Asset-specific cycle parameters
        self.asset_type = asset_type
        self.cycle_params = {
            'BITCOIN': {'daily': (54, 66), 'weekly': (24, 42), 'yearly': 4},  # From PDF
            'ETHEREUM': {'daily': (54, 66), 'weekly': (24, 42), 'yearly': 4},  # From PDF
            'SP500': {'daily': (36, 44), 'weekly': (22, 31), 'yearly': 4},  # From PDF
            'NASDAQ': {'daily': (36, 44), 'weekly': (22, 31), 'yearly': 4},  # From PDF
            'DOW': {'daily': (36, 44), 'weekly': (22, 31), 'yearly': 4},  # From PDF
            'GOLD': {'daily': (22, 28), 'weekly': (22, 26), 'yearly': 8},  # From PDF
            'OIL': {'daily': (36, 44), 'weekly': (22, 26), 'yearly': 3},  # From PDF
            'DXY': {'daily': (15, 24), 'weekly': (16, 20), 'yearly': 3}  # From PDF
        }
        
        # Update with custom parameters if provided
        if custom_params:
            self.cycle_params.update(custom_params)
        
        # Initialize CMA strategy settings
        self.use_cma_strategy = False  # Default to off
        
        # Calculate various SMAs for cycle confirmation
        self.data['SMA10'] = calculate_sma(self.data['Close'], window=10)
        self.data['SMA20'] = calculate_sma(self.data['Close'], window=20)
        self.data['SMA50'] = calculate_sma(self.data['Close'], window=50)
        self.data['SMA200'] = calculate_sma(self.data['Close'], window=200)
        
        # Calculate RSI for cycle confirmation
        self.data['RSI'] = self.calculate_rsi(self.data['Close'], window=14)
        
        # Calculate volume SMA and momentum
        if 'Volume' in self.data.columns:
            self.data['Volume_SMA20'] = calculate_sma(self.data['Volume'], window=20)
            self.data['Volume_Momentum'] = self.data['Volume'] / self.data['Volume_SMA20']
        
        # Initialize cycle properties
        self.cycles = {
            'daily': {'name': 'Daily Cycle'},
            'weekly': {'name': 'Weekly Cycle'},
            'yearly': {'name': 'Yearly Cycle'}
        }
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def detect_cycle_lows(self, timeframe='daily'):
        """
        Detect cycle lows based on Bob/Camel's rules and asset-specific timing windows.
        Returns a list of cycle low indices.
        """
        if timeframe == 'daily':
            min_days, max_days = self.cycle_params[self.asset_type]['daily']
            window = min_days // 5  # Even smaller window for better sensitivity
            ma_reference = 'SMA10'
            rsi_threshold = 35
            # Extended windows for early/late cycles
            extended_min = min_days * 0.5  # Allow even shorter cycles
            extended_max = max_days * 1.4  # Allow cycles as long as 140% of normal
        elif timeframe == 'weekly':
            min_days, max_days = self.cycle_params[self.asset_type]['weekly']
            window = min_days // 3
            ma_reference = 'SMA20'
            rsi_threshold = 40
            extended_min = min_days * 0.7
            extended_max = max_days * 1.3
        else:  # yearly
            min_days = self.cycle_params[self.asset_type]['yearly'] * 365
            max_days = min_days * 1.2
            window = min_days // 3
            ma_reference = 'SMA200'
            rsi_threshold = 40
            extended_min = min_days * 0.8
            extended_max = max_days * 1.2
        
        # Use swing low detection with appropriate window
        potential_lows = detect_swing_lows(self.data['Close'], window=window)
        
        # Filter cycle lows based on Bob/Camel's rules
        filtered_lows = []
        for i in range(len(potential_lows)-1):
            current_low = potential_lows[i]
            next_low = potential_lows[i+1]
            
            # Get the dates from the index
            current_date = self.data.index[current_low]
            next_date = self.data.index[next_low]
            
            # Calculate the difference in days/weeks
            if timeframe == 'weekly':
                periods_between = (next_date - current_date).days / 7
            else:
                periods_between = (next_date - current_date).days
            
            # Check if this is a valid cycle low
            price = self.data['Close'].iloc[current_low]
            ma_value = self.data[ma_reference].iloc[current_low]
            rsi_value = self.data['RSI'].iloc[current_low]
            
            # Calculate scores for different conditions
            timing_score = 2 if min_days <= periods_between <= max_days else \
                         1 if extended_min <= periods_between <= extended_max else 0
            
            price_ma_score = 2 if price <= ma_value else \
                           1 if price <= ma_value * 1.02 else 0
            
            rsi_score = 2 if rsi_value < rsi_threshold else \
                       1 if rsi_value < rsi_threshold + 5 else 0
            
            # Check for higher high after the low (within next 15 periods)
            future_slice = slice(current_low + 1, min(current_low + 16, len(self.data)))
            future_high = self.data['High'].iloc[future_slice].max()  # Use High instead of Close
            future_low = self.data['Low'].iloc[future_slice].min()  # Check if price moves lower first
            
            # Score based on price movement pattern - more lenient
            higher_high_score = 2 if future_high > price * 1.03 else \
                              1 if future_high > price * 1.01 else 0
            
            # Volume condition (if available)
            volume_score = 0
            if 'Volume' in self.data.columns:
                vol_ratio = self.data['Volume'].iloc[current_low] / self.data['Volume_SMA20'].iloc[current_low]
                volume_score = 2 if vol_ratio > 1.2 else \
                             1 if vol_ratio > 0.8 else 0
            
            # Calculate total score
            total_score = timing_score + price_ma_score + rsi_score + higher_high_score + volume_score
            min_required_score = 3 if timing_score > 0 else 4  # Lower threshold for all cycles
            
            if total_score >= min_required_score:
                filtered_lows.append(current_low)
        
        # Add the last cycle low if it meets criteria
        if len(potential_lows) > 0:
            last_low = potential_lows[-1]
            last_date = self.data.index[last_low]
            
            if timeframe == 'weekly':
                periods_since_last = (self.data.index[-1] - last_date).days / 7
            else:
                periods_since_last = (self.data.index[-1] - last_date).days
            
            # Check if this could be the start of a new cycle - more lenient for recent lows
            if periods_since_last >= extended_min * 0.4:  # Even more lenient for last low
                price = self.data['Close'].iloc[last_low]
                ma_value = self.data[ma_reference].iloc[last_low]
                rsi_value = self.data['RSI'].iloc[last_low]
                
                # Calculate scores for the last low
                timing_score = 1  # Give some credit for timing
                price_ma_score = 2 if price <= ma_value else \
                               1 if price <= ma_value * 1.03 else 0  # More lenient MA comparison
                rsi_score = 2 if rsi_value < rsi_threshold else \
                           1 if rsi_value < rsi_threshold + 7 else 0  # More lenient RSI
                
                # Volume score
                volume_score = 0
                if 'Volume' in self.data.columns:
                    vol_ratio = self.data['Volume'].iloc[last_low] / self.data['Volume_SMA20'].iloc[last_low]
                    volume_score = 2 if vol_ratio > 1.1 else \
                                 1 if vol_ratio > 0.7 else 0  # More lenient volume requirement
                
                total_score = timing_score + price_ma_score + rsi_score + volume_score
                
                if total_score >= 2:  # Keep low threshold for last cycle low
                    filtered_lows.append(last_low)
        
        return filtered_lows
    
    def identify_cycle_translation(self, cycle_start, cycle_end):
        """
        Identify if a cycle is Left Translated, Mid Translated, or Right Translated.
        """
        # Get the cycle data
        cycle_data = self.data['Close'].iloc[cycle_start:cycle_end+1]
        cycle_length = len(cycle_data)
        mid_point = cycle_length // 2
        
        # Find the position of the high point within the cycle
        high_point_position = cycle_data.values.argmax()
        
        if high_point_position < mid_point:
            return 'Left Translated'
        elif high_point_position > mid_point:
            return 'Right Translated'
        else:
            return 'Mid Translated'
    
    def detect_cycle_failure(self, cycle_start, cycle_end):
        """
        Detect if a cycle has failed (makes a lower low than the cycle low).
        A cycle fails if price makes a new low below the cycle low before making a higher high.
        """
        if cycle_start == 0:
            return False
        
        cycle_data = self.data.iloc[cycle_start:cycle_end+1]
        cycle_low = cycle_data['Low'].iloc[0]  # The initial low of the cycle
        cycle_high = cycle_data['High'].max()  # The highest point in the cycle
        high_idx = cycle_data['High'].idxmax()  # When the high occurred
        
        # Check if price made a lower low after the cycle low but before the cycle high
        data_before_high = cycle_data.loc[:high_idx]
        if len(data_before_high) > 1:  # Make sure we have data to check
            lowest_before_high = data_before_high['Low'].min()
            return lowest_before_high < cycle_low
        
        return False
    
    def detect_half_cycle_low(self, cycle_start, cycle_end):
        """
        Detect half cycle lows and their characteristics.
        """
        cycle_data = self.data['Close'].iloc[cycle_start:cycle_end+1]
        mid_point = cycle_start + (cycle_end - cycle_start) // 2
        
        # Look for lows around the midpoint
        half_cycle_window = 5  # days
        half_cycle_data = cycle_data.iloc[mid_point-half_cycle_window:mid_point+half_cycle_window]
        
        if len(half_cycle_data) > 0:
            half_cycle_low = half_cycle_data.min()
            half_cycle_low_idx = half_cycle_data.idxmin()
            
            # Compare with prior cycle low
            prior_cycle_low = self.data['Close'].iloc[cycle_start-1] if cycle_start > 0 else None
            
            if prior_cycle_low is not None:
                is_bullish = half_cycle_low > prior_cycle_low
            else:
                is_bullish = True
            
            return {
                'exists': True,
                'index': half_cycle_low_idx,
                'price': half_cycle_low,
                'is_bullish': is_bullish
            }
        
        return {'exists': False}
    
    def detect_cma_cycles(self, timeframe='daily'):
        """
        Detect cycle lows using the centered moving average (CMA) strategy.
        This is an alternative method that can be used alongside the main cycle detection.
        """
        if timeframe == 'daily':
            min_days, max_days = self.cycle_params[self.asset_type]['daily']
            cma_period = (min_days + max_days) // 2  # Use average cycle length
            ma_reference = 'SMA20'
        elif timeframe == 'weekly':
            min_days, max_days = self.cycle_params[self.asset_type]['weekly']
            cma_period = ((min_days + max_days) // 2) * 7  # Convert weeks to days
            ma_reference = 'SMA50'
        else:
            return []  # CMA strategy not implemented for yearly cycles
        
        # Calculate centered moving average
        half_period = cma_period // 2
        cma = self.data['Close'].rolling(window=cma_period, center=True).mean()
        self.data['CMA'] = cma  # Store for plotting if needed
        
        # Find potential cycle lows
        potential_lows = []
        for i in range(half_period, len(self.data) - half_period):
            price = self.data['Close'].iloc[i]
            price_prev = self.data['Close'].iloc[i-1]
            price_next = self.data['Close'].iloc[i+1]
            cma_value = cma.iloc[i]
            
            # Check if this is a local minimum
            if price <= price_prev and price <= price_next:
                # Check if price is below CMA
                if price < cma_value:
                    # Check if RSI is showing oversold
                    rsi = self.data['RSI'].iloc[i]
                    if rsi < 40:  # More lenient RSI threshold for CMA strategy
                        potential_lows.append(i)
        
        # Filter lows based on minimum distance
        filtered_lows = []
        min_distance = min_days // 2  # Allow for shorter cycles in CMA strategy
        
        for i, low in enumerate(potential_lows):
            if i == 0:
                filtered_lows.append(low)
                continue
                
            prev_low = filtered_lows[-1]
            days_between = (self.data.index[low] - self.data.index[prev_low]).days
            
            if days_between >= min_distance:
                filtered_lows.append(low)
        
        return filtered_lows

    def analyze_cycles(self):
        """
        Perform comprehensive cycle analysis across all timeframes.
        """
        analysis = {}
        
        # First detect daily cycles
        daily_cycles = []
        if self.use_cma_strategy:
            daily_lows = self.detect_cma_cycles('daily')
        else:
            daily_lows = self.detect_cycle_lows('daily')
        
        for i in range(len(daily_lows)-1):
            start = daily_lows[i]
            end = daily_lows[i+1]
            
            cycle_info = {
                'start': start,
                'end': end,
                'translation': self.identify_cycle_translation(start, end),
                'failed': self.detect_cycle_failure(start, end),
                'half_cycle': self.detect_half_cycle_low(start, end)
            }
            daily_cycles.append(cycle_info)
        
        analysis['daily'] = daily_cycles
        
        # Then detect weekly cycles
        weekly_cycles = []
        if self.use_cma_strategy:
            weekly_lows = self.detect_cma_cycles('weekly')
        else:
            weekly_lows = self.detect_cycle_lows('weekly')
        
        for i in range(len(weekly_lows)-1):
            start = weekly_lows[i]
            end = weekly_lows[i+1]
            
            cycle_info = {
                'start': start,
                'end': end,
                'translation': self.identify_cycle_translation(start, end),
                'failed': self.detect_cycle_failure(start, end),
                'half_cycle': self.detect_half_cycle_low(start, end)
            }
            weekly_cycles.append(cycle_info)
        
        analysis['weekly'] = weekly_cycles
        
        # Finally detect yearly cycles (always use standard method)
        yearly_cycles = []
        yearly_lows = self.detect_cycle_lows('yearly')
        
        for i in range(len(yearly_lows)-1):
            start = yearly_lows[i]
            end = yearly_lows[i+1]
            
            cycle_info = {
                'start': start,
                'end': end,
                'translation': self.identify_cycle_translation(start, end),
                'failed': self.detect_cycle_failure(start, end),
                'half_cycle': self.detect_half_cycle_low(start, end)
            }
            yearly_cycles.append(cycle_info)
        
        analysis['yearly'] = yearly_cycles
        
        return analysis
    
    def plot_cycle_analysis(self, start_date=None, end_date=None):
        """
        Create a comprehensive visualization of the cycle analysis.
        """
        if start_date is None:
            start_date = self.data.index[0]
        if end_date is None:
            end_date = self.data.index[-1]
            
        mask = (self.data.index >= start_date) & (self.data.index <= end_date)
        plot_data = self.data[mask]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
        
        # Plot price and SMAs
        ax1.plot(plot_data.index, plot_data['Close'], label='Price', color='blue')
        ax1.plot(plot_data.index, plot_data['SMA20'], label='20 SMA', color='orange')
        ax1.plot(plot_data.index, plot_data['SMA50'], label='50 SMA', color='red')
        ax1.plot(plot_data.index, plot_data['SMA200'], label='200 SMA', color='purple')
        
        # Plot volume if available
        if 'Volume' in plot_data.columns:
            ax2.bar(plot_data.index, plot_data['Volume'], label='Volume', color='gray', alpha=0.5)
            ax2.plot(plot_data.index, plot_data['Volume_SMA20'], label='Volume SMA20', color='red')
        
        # Analyze and plot cycles
        analysis = self.analyze_cycles()
        
        # Plot daily cycles
        for cycle in analysis['daily']:
            start = cycle['start']
            end = cycle['end']
            
            # Color based on translation
            color = {
                'Left Translated': 'red',
                'Right Translated': 'green',
                'Mid Translated': 'yellow'
            }.get(cycle['translation'])
            
            # Plot cycle region
            ax1.axvspan(plot_data.index[start], plot_data.index[end], color=color, alpha=0.1)
            
            # Plot cycle low
            ax1.scatter(plot_data.index[start], plot_data['Close'].iloc[start],
                       color='red', zorder=5)
            
            # Plot half cycle low if exists
            if cycle['half_cycle']['exists']:
                half_cycle_idx = cycle['half_cycle']['index']
                half_cycle_color = 'green' if cycle['half_cycle']['is_bullish'] else 'red'
                ax1.scatter(plot_data.index[half_cycle_idx], cycle['half_cycle']['price'],
                          color=half_cycle_color, marker='^', zorder=5)
            
            # Add translation label
            if cycle['failed']:
                ax1.text(plot_data.index[start], plot_data['Close'].max(),
                        f"{cycle['translation']} (Failed)", rotation=45, color=color)
            else:
                ax1.text(plot_data.index[start], plot_data['Close'].max(),
                        cycle['translation'], rotation=45, color=color)
        
        ax1.set_title(f'{self.asset_type} Cycle Analysis')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_title('Volume Analysis')
        ax2.set_ylabel('Volume')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    # Example usage
    filename = input("Enter CSV filename containing Date and Close columns: ")
    asset_type = input("Enter asset type (BITCOIN, ETHEREUM, SP500, NASDAQ, DOW, GOLD, OIL, DXY): ")
    
    try:
        data = pd.read_csv(filename)
        analyzer = BressartCycleIndicator(data, asset_type)
        
        # Plot the analysis
        analyzer.plot_cycle_analysis()
        
    except Exception as e:
        print("Error:", e)

if __name__ == '__main__':
    main() 