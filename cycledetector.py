import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_sma(series, window=10):
    """
    Calculate a simple moving average (SMA) over the given window.
    """
    return series.rolling(window=window).mean()

def detect_swing_lows(prices, window=3):
    """
    Identify swing lows in the price series.
    
    A swing low is defined as a point which is the minimum within a window of 
    (window) bars before and after it. Adjust 'window' to tune sensitivity.
    Returns a list of indices corresponding to swing lows.
    """
    lows = []
    # Loop over the price series avoiding the edges.
    for i in range(window, len(prices) - window):
        # If the price is the minimum among the (window*2 + 1) points centered at i,
        # we mark it as a swing low.
        if prices[i] == prices[i-window:i+window+1].min():
            lows.append(i)
    return lows

def analyze_cycles(prices, swing_lows):
    """
    Analyze cycles defined by consecutive swing lows.
    
    For each cycle (from one swing low to the next), the function:
      • Finds the highest price point in that cycle (the cycle high).
      • Computes the time midpoint between the two swing lows.
      • Classifies the cycle based on the timing of the high:
          - If the high occurs before the midpoint, it is Left Translated.
          - If after the midpoint, Right Translated.
          - If near the midpoint, Mid Translated.
    
    Returns a list of dictionaries with cycle details.
    """
    cycles = []
    for i in range(len(swing_lows) - 1):
        start = swing_lows[i]
        end = swing_lows[i+1]
        cycle_data = prices[start:end+1]
        # Get the index (within the overall series) of the cycle high
        relative_high_index = cycle_data.idxmax()
        mid_point = (start + end) / 2.0
        # Determine translation type by comparing the index of the high with the midpoint.
        if relative_high_index < mid_point:
            translation = 'Left Translated'
        elif relative_high_index > mid_point:
            translation = 'Right Translated'
        else:
            translation = 'Mid Translated'
        cycles.append({
            'start': start,
            'end': end,
            'high': relative_high_index,
            'translation': translation
        })
    return cycles

def plot_cycles(data, swing_lows, cycles, sma=None):
    """
    Plot the price data, overlaying swing lows, cycle highs, and cycle zones.
    
    If an SMA series is provided, it will be plotted for confirmation signals.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Close'], label='Price', color='blue')
    
    # Plot the SMA if provided
    if sma is not None:
        plt.plot(data.index, sma, label='10-Period SMA', color='magenta', linestyle='--')
    
    # Plot swing lows
    plt.scatter(data.index[swing_lows], data['Close'].iloc[swing_lows],
                color='red', label='Swing Lows', zorder=5)
    
    # Plot each cycle's high and annotate with its translation type.
    for cycle in cycles:
        start = cycle['start']
        end = cycle['end']
        high = cycle['high']
        plt.scatter(data.index[high], data['Close'].iloc[high],
                    color='green', label='Cycle High', zorder=5)
        plt.annotate(cycle['translation'],
                     xy=(data.index[high], data['Close'].iloc[high]),
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', fontsize=9, color='darkgreen')
        # Shade the cycle region to help visualize the cycle low-to-low segment.
        plt.axvspan(data.index[start], data.index[end], color='yellow', alpha=0.1)
    
    plt.title("Trading Cycle Analysis")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # Load price data from a CSV file.
    # The CSV should contain at least two columns: 'Date' and 'Close'.
    filename = input("Enter CSV filename containing Date and Close columns (e.g., data.csv): ")
    try:
        data = pd.read_csv(filename, parse_dates=['Date'])
    except Exception as e:
        print("Error loading file:", e)
        return

    # Ensure data is sorted by date and set Date as index.
    data.sort_values('Date', inplace=True)
    data.set_index('Date', inplace=True)
    
    # Calculate a 10-day SMA for confirmation purposes.
    data['SMA10'] = calculate_sma(data['Close'], window=10)
    
    # Detect swing lows using a rolling window (adjust window parameter if necessary)
    window_param = int(input("Enter window size for swing low detection (default is 3): ") or 3)
    swing_lows = detect_swing_lows(data['Close'], window=window_param)
    
    if len(swing_lows) < 2:
        print("Not enough swing lows detected. Try lowering the window parameter or check your data.")
        return

    # Analyze cycles (from swing low to swing low)
    cycles = analyze_cycles(data['Close'], swing_lows)
    
    # Plot the cycles, swing lows, cycle highs, and the SMA for confirmation
    plot_cycles(data, swing_lows, cycles, sma=data['SMA10'])
    
    # Allow the user to confirm the detection
    confirm = input("Are the identified cycles (and their classifications) correct? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cycle analysis did not meet expectations. You may adjust parameters (e.g., swing low window) and rerun the script.")
    else:
        print("Cycle analysis confirmed.")
    
if __name__ == '__main__':
    main()
