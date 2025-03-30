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
        
        # Calculate various SMAs for cycle confirmation
        self.data['SMA10'] = calculate_sma(self.data['Close'], window=10)
        self.data['SMA20'] = calculate_sma(self.data['Close'], window=20)
        self.data['SMA50'] = calculate_sma(self.data['Close'], window=50)
        self.data['SMA200'] = calculate_sma(self.data['Close'], window=200)
        
        # Calculate RSI for cycle confirmation
        self.data['RSI'] = self.calculate_rsi(self.data['Close'], window=14)
        self.data['RSI3M3'] = self.calculate_rsi3m3(self.data['Close'])
        
        # Calculate Stochastic for cycle confirmation
        self.data['Stoch_K'], self.data['Stoch_D'] = self.calculate_stochastic(self.data['High'], 
                                                                              self.data['Low'], 
                                                                              self.data['Close'])
        
        # Calculate MACD for trend confirmation
        self.data['MACD'], self.data['MACD_Signal'], self.data['MACD_Hist'] = self.calculate_macd(self.data['Close'])
        
        # Calculate Bollinger Bands
        self.data['BB_Middle'], self.data['BB_Upper'], self.data['BB_Lower'] = self.calculate_bollinger_bands(self.data['Close'])
        
        # Calculate volume indicators
        if 'Volume' in self.data.columns:
            self.data['Volume_SMA20'] = calculate_sma(self.data['Volume'], window=20)
            self.data['Volume_Momentum'] = self.data['Volume'] / self.data['Volume_SMA20']
            self.data['OBV'] = self.calculate_obv(self.data['Close'], self.data['Volume'])
            self.data['CMF'] = self.calculate_cmf(self.data['High'], self.data['Low'], 
                                                 self.data['Close'], self.data['Volume'])
        
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
    
    def calculate_centered_ma(self, prices, period=20):
        """
        Calculate centered moving average as described in Bressert's manual.
        The MA is plotted back half the period from the most recent close.
        """
        # Calculate regular moving average
        ma = prices.rolling(window=period).mean()
        # Shift it back by half the period (10 days for 20-bar MA)
        centered_ma = ma.shift(-period//2)
        return centered_ma

    def calculate_rsi3m3(self, prices):
        """
        Calculate RSI3M3 as described in Bressert's manual:
        RSI with period 3 smoothed with 3-bar moving average
        """
        # Calculate RSI with period 3
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=3).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=3).mean()
        rs = gain / loss
        rsi3 = 100 - (100 / (1 + rs))
        # Smooth with 3-bar moving average
        rsi3m3 = rsi3.rolling(window=3).mean()
        return rsi3m3

    def calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()
        
        return k, d

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator."""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return macd, signal_line, histogram

    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands."""
        middle_band = prices.rolling(window=window).mean()
        std_dev = prices.rolling(window=window).std()
        upper_band = middle_band + (std_dev * num_std)
        lower_band = middle_band - (std_dev * num_std)
        
        return middle_band, upper_band, lower_band

    def calculate_obv(self, close, volume):
        """Calculate On-Balance Volume."""
        price_change = close.diff()
        obv = (volume * (price_change > 0).astype(int) - 
               volume * (price_change < 0).astype(int)).cumsum()
        return obv

    def calculate_cmf(self, high, low, close, volume, period=20):
        """Calculate Chaikin Money Flow."""
        mfm = ((close - low) - (high - close)) / (high - low)
        mfv = mfm * volume
        cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
        return cmf

    def get_cycle_confirmation_signals(self, index):
        """
        Get comprehensive confirmation signals for a potential cycle low.
        Returns a dictionary of technical signals and their values.
        """
        signals = {
            'price_action': {
                'below_20ma': self.data['Close'].iloc[index] < self.data['SMA20'].iloc[index],
                'below_50ma': self.data['Close'].iloc[index] < self.data['SMA50'].iloc[index],
                'below_bb_lower': self.data['Close'].iloc[index] < self.data['BB_Lower'].iloc[index]
            },
            'momentum': {
                'rsi': self.data['RSI'].iloc[index],
                'rsi_oversold': self.data['RSI'].iloc[index] < 30,
                'rsi3m3': self.data['RSI3M3'].iloc[index],
                'rsi3m3_oversold': self.data['RSI3M3'].iloc[index] < 30,
                'stoch_k': self.data['Stoch_K'].iloc[index],
                'stoch_d': self.data['Stoch_D'].iloc[index],
                'stoch_oversold': self.data['Stoch_K'].iloc[index] < 20
            },
            'trend': {
                'macd': self.data['MACD'].iloc[index],
                'macd_signal': self.data['MACD_Signal'].iloc[index],
                'macd_hist': self.data['MACD_Hist'].iloc[index],
                'macd_bullish_cross': (index > 0 and 
                                     self.data['MACD_Hist'].iloc[index-1] < 0 and 
                                     self.data['MACD_Hist'].iloc[index] > 0)
            }
        }
        
        if 'Volume' in self.data.columns:
            signals['volume'] = {
                'volume_spike': self.data['Volume_Momentum'].iloc[index] > 1.2,
                'obv_rising': self.data['OBV'].iloc[index] > self.data['OBV'].iloc[index-1],
                'cmf': self.data['CMF'].iloc[index],
                'money_flow_positive': self.data['CMF'].iloc[index] > 0
            }
        
        # Calculate confirmation score (0-100)
        score = 0
        score += 20 if signals['price_action']['below_20ma'] else 0
        score += 20 if signals['momentum']['rsi_oversold'] else 0
        score += 20 if signals['momentum']['stoch_oversold'] else 0
        score += 20 if signals['trend']['macd_bullish_cross'] else 0
        score += 20 if 'volume' in signals and signals['volume']['volume_spike'] else 0
        
        signals['confirmation_score'] = score
        
        return signals

    def detect_cycle_lows(self, timeframe='daily'):
        """
        Detect cycle lows using RSI3M3 strategy with Bressert's methodology.
        """
        # Get cycle parameters based on timeframe
        if timeframe == 'daily':
            min_days, max_days = self.cycle_params[self.asset_type]['daily']
        else:
            min_weeks, max_weeks = self.cycle_params[self.asset_type]['weekly']
            min_days, max_days = min_weeks * 7, max_weeks * 7
        
        # Calculate minimum days between lows (40% of minimum cycle length)
        min_distance = int(min_days * 0.4)
        
        potential_lows = []
        last_low_idx = None
        
        for i in range(3, len(self.data)-1):
            # Skip if too close to previous low
            if last_low_idx is not None and i - last_low_idx < min_distance:
                continue
            
            # Check for local price minimum
            price_series = self.data['Low']
            if (price_series.iloc[i] <= price_series.iloc[i-1] and 
                price_series.iloc[i] <= price_series.iloc[i+1]):
                
                # Get confirmation signals
                signals = self.get_cycle_confirmation_signals(i)
                score = 0
                explanation = []
                
                # 1. RSI3M3 Oversold (0-30 points)
                rsi3m3 = self.data['RSI3M3'].iloc[i]
                if rsi3m3 < 30:
                    score += 30
                    explanation.append(f"RSI3M3 oversold at {rsi3m3:.1f}")
                elif rsi3m3 < 35:
                    score += 20
                    explanation.append(f"RSI3M3 near oversold at {rsi3m3:.1f}")
                
                # 2. Price Action (0-20 points)
                if signals['price_action']['below_20ma']:
                    score += 10
                    explanation.append("Price below 20MA")
                if signals['price_action']['below_50ma']:
                    score += 10
                    explanation.append("Price below 50MA")
                
                # 3. Momentum Confirmation (0-20 points)
                if signals['momentum']['rsi_oversold']:
                    score += 10
                    explanation.append(f"RSI oversold at {signals['momentum']['rsi']:.1f}")
                if signals['momentum']['stoch_oversold']:
                    score += 10
                    explanation.append(f"Stochastic oversold at {signals['momentum']['stoch_k']:.1f}")
                
                # 4. Volume Confirmation (0-15 points)
                if 'volume' in signals:
                    if signals['volume']['volume_spike']:
                        score += 10
                        explanation.append("Volume spike detected")
                    if signals['volume']['obv_rising']:
                        score += 5
                        explanation.append("OBV rising")
                
                # 5. MACD Confirmation (0-15 points)
                if signals['trend']['macd_bullish_cross']:
                    score += 15
                    explanation.append("MACD bullish cross")
                elif signals['trend']['macd_hist'] > signals['trend']['macd_hist']:
                    score += 5
                    explanation.append("MACD histogram improving")
                
                # Check timing if we have a previous low
                if last_low_idx is not None:
                    days_since_last = (self.data.index[i] - self.data.index[last_low_idx]).days
                    if min_days * 0.8 <= days_since_last <= max_days * 1.2:
                        score += 10
                        explanation.append(f"Good timing: {days_since_last} days since last low")
                
                # Accept if score is sufficient (45+ points)
                if score >= 45:
                    potential_lows.append({
                        'index': i,
                        'signal_date': self.data.index[i],
                        'confirmation_signals': signals,
                        'confirmation_score': score,
                        'explanation': "\n".join(explanation)
                    })
                    last_low_idx = i
        
        return potential_lows

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
                        potential_lows.append({
                            'index': i,
                            'signal_date': self.data.index[i]
                        })
        
        # Filter lows based on minimum distance
        filtered_lows = []
        min_distance = min_days // 2  # Allow for shorter cycles in CMA strategy
        
        for i, low in enumerate(potential_lows):
            if i == 0:
                filtered_lows.append(low)
                continue
                
            prev_low = filtered_lows[-1]['index']
            days_between = (self.data.index[low['index']] - self.data.index[prev_low]).days
            
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
        daily_lows = self.detect_cycle_lows('daily')
        
        for i in range(len(daily_lows)-1):
            start = daily_lows[i]['index']
            end = daily_lows[i+1]['index']
            signal_date = daily_lows[i]['signal_date']
            
            cycle_info = {
                'start': start,
                'end': end,
                'signal_date': signal_date,
                'translation': self.identify_cycle_translation(start, end),
                'failed': self.detect_cycle_failure(start, end),
                'half_cycle': self.detect_half_cycle_low(start, end)
            }
            daily_cycles.append(cycle_info)
        
        analysis['daily'] = daily_cycles
        
        # Then detect weekly cycles
        weekly_cycles = []
        weekly_lows = self.detect_cycle_lows('weekly')
        
        for i in range(len(weekly_lows)-1):
            start = weekly_lows[i]['index']
            end = weekly_lows[i+1]['index']
            signal_date = weekly_lows[i]['signal_date']
            
            cycle_info = {
                'start': start,
                'end': end,
                'signal_date': signal_date,
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
            start = yearly_lows[i]['index']
            end = yearly_lows[i+1]['index']
            signal_date = yearly_lows[i]['signal_date']
            
            cycle_info = {
                'start': start,
                'end': end,
                'signal_date': signal_date,
                'translation': self.identify_cycle_translation(start, end),
                'failed': self.detect_cycle_failure(start, end),
                'half_cycle': self.detect_half_cycle_low(start, end)
            }
            yearly_cycles.append(cycle_info)
        
        analysis['yearly'] = yearly_cycles
        
        return analysis
    
    def calculate_next_low_window(self, last_low_date, timeframe='daily'):
        """
        Calculate the expected window for the next cycle low based on the last low.
        Returns tuple of (earliest_date, latest_date, normal_earliest, normal_latest)
        """
        if timeframe == 'daily':
            min_days, max_days = self.cycle_params[self.asset_type]['daily']
            expected_length = (min_days + max_days) / 2
            normal_variance = expected_length * 0.15  # 15% variance for normal cases
            extended_variance = expected_length * 0.30  # 30% variance for exceptional cases
        else:
            min_weeks, max_weeks = self.cycle_params[self.asset_type]['weekly']
            expected_length = (min_weeks + max_weeks) / 2 * 7
            normal_variance = expected_length * 0.15
            extended_variance = expected_length * 0.30
        
        # Calculate normal window
        normal_earliest = last_low_date + pd.Timedelta(days=int(expected_length - normal_variance))
        normal_latest = last_low_date + pd.Timedelta(days=int(expected_length + normal_variance))
        
        # Calculate extended window
        extended_earliest = last_low_date + pd.Timedelta(days=int(expected_length - extended_variance))
        extended_latest = last_low_date + pd.Timedelta(days=int(expected_length + extended_variance))
        
        return (extended_earliest, extended_latest, normal_earliest, normal_latest)

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
        
        # Plot daily cycles and next low windows
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
            
            # Calculate and plot next low window
            if plot_data.index[start] < end_date:  # Only plot if the low is within our view
                extended_earliest, extended_latest, normal_earliest, normal_latest = self.calculate_next_low_window(
                    plot_data.index[start]
                )
                
                # Plot extended window with light shading
                if extended_earliest <= end_date:
                    ax1.axvspan(extended_earliest, extended_latest,
                              color='gray', alpha=0.1,
                              label='Extended Window' if start == analysis['daily'][0]['start'] else "")
                
                # Plot normal window with darker shading
                if normal_earliest <= end_date:
                    ax1.axvspan(normal_earliest, normal_latest,
                              color='gray', alpha=0.2,
                              label='Normal Window' if start == analysis['daily'][0]['start'] else "")
                
                # Add text for window dates
                y_pos = plot_data['Close'].min()
                ax1.text(normal_earliest, y_pos,
                        f'Window Start\n{normal_earliest.strftime("%Y-%m-%d")}',
                        rotation=45, verticalalignment='top')
                ax1.text(normal_latest, y_pos,
                        f'Window End\n{normal_latest.strftime("%Y-%m-%d")}',
                        rotation=45, verticalalignment='top')
        
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

    def print_debug_info(self, start_date, end_date):
        """Print detailed debug information for potential cycle lows in the specified date range."""
        mask = (self.data.index >= start_date) & (self.data.index <= end_date)
        data_slice = self.data[mask]
        
        for i in range(len(data_slice)):
            idx = data_slice.index[i]
            price = self.data['Close'].loc[idx]
            rsi3m3 = self.calculate_rsi3m3(self.data['Close']).loc[idx]
            signals = self.get_cycle_confirmation_signals(self.data.index.get_loc(idx))
            
            # Only print info for potential lows (RSI3M3 < 35 or price below 20MA)
            if rsi3m3 < 35 or price < self.data['SMA20'].loc[idx]:
                print(f"\nDate: {idx.strftime('%Y-%m-%d')}")
                print(f"Price: ${price:.2f}")
                print(f"RSI3M3: {rsi3m3:.1f}")
                print(f"Below 20MA: {'Yes' if price < self.data['SMA20'].loc[idx] else 'No'}")
                print(f"Below 50MA: {'Yes' if price < self.data['SMA50'].loc[idx] else 'No'}")
                if 'Volume' in self.data.columns:
                    vol_ratio = self.data['Volume'].loc[idx] / self.data['Volume'].loc[idx-5:idx].mean()
                    print(f"Volume vs 5-day avg: {vol_ratio:.1f}x")
                print(f"MACD Histogram: {self.data['MACD_Hist'].loc[idx]:.3f}")
                print("-" * 40)

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