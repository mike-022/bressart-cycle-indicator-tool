# Bressart Cycle Indicator Tool

A powerful market cycle analysis tool that helps traders and investors identify market cycles across different timeframes and assets.

## Features

### 1. Multi-Asset Support
- Pre-configured assets with optimized cycle parameters:
  - Cryptocurrencies: Bitcoin (54-66 days), Ethereum (54-66 days)
  - Indices: S&P 500 (36-44 days), NASDAQ (36-44 days), DOW (36-44 days)
  - Commodities: Gold (22-28 days), Oil (36-44 days)
  - Currencies: DXY (15-24 days)
- Custom asset support with configurable cycle parameters

### 2. Multiple Timeframes
- Daily cycle analysis
- Weekly cycle analysis
- Yearly cycle tracking
- Custom date range selection

### 3. Advanced Cycle Detection
- Two detection methods:
  - Standard cycle detection using multiple indicators
  - Centered Moving Average (CMA) strategy (recommended for custom assets)
- Cycle translation identification (Left/Right/Mid)
- Failed cycle detection
- Automatic timing window projection

### 4. Technical Indicators
- Moving Averages (20 SMA, 50 SMA)
- Volume analysis with 20-period SMA
- RSI for cycle confirmation
- Custom indicator thresholds for each asset

### 5. Interactive Visualization
- Candlestick charts with cycle markers
- Volume analysis panel
- Cycle high/low annotations
- Future cycle timing windows
- Detailed cycle statistics

## Example

![Bressart Cycle Indicator Tool Screenshot](docs/images/tool_screenshot.png)

The screenshot above demonstrates the tool's clean interface analyzing Bitcoin's daily cycles:

**Chart Elements:**
- Price action with candlestick patterns
- Key moving averages:
  - 20 SMA (orange)
  - 50 SMA (red)
- Cycle markers:
  - Daily Cycle Lows (DCL): Green triangles up
  - Daily Cycle Highs (DCH): Green triangles down
- Next cycle timing window (yellow box)
- Volume analysis with 20-period SMA

**Interface Features:**
- Clean, dark theme design
- Essential indicators only in legend
- Clear cycle annotations
- Interactive sidebar controls for:
  - Asset selection
  - Timeframe choice
  - Cycle detection method
  - Custom date ranges

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bressart-cycle-indicator.git
cd bressart-cycle-indicator
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

### Standard Assets
1. Select an asset from the dropdown menu
2. Choose your preferred timeframe (daily/weekly)
3. Select the time period (1y, 2y, 5y, max)
4. Optional: Enable CMA strategy
5. Optional: Set custom date range

### Custom Assets
1. Enable "Use Custom Asset" checkbox
2. Enter the asset symbol (e.g., AAPL, MSFT)
3. Configure cycle parameters:
   - Daily cycle min/max days
   - Weekly cycle min/max weeks
   - Yearly cycle count
4. The CMA strategy is automatically enabled for optimal results

### Reading the Results

#### Cycle Markers
- Daily Cycles:
  - Green triangles: Cycle lows (DCL)
  - Red triangles: Cycle highs (DCH)
- Weekly Cycles:
  - Purple diamonds: Cycle lows (WCL)
  - Purple triangles: Cycle highs (WCH)

#### Cycle Statistics
For each cycle, you'll see:
- Duration (days/weeks)
- Low price and date
- High price and date
- Return percentage
- Translation type (Left/Right/Mid)
- Status (Valid/Failed)

#### Timing Windows
Yellow shaded areas indicate projected timing windows for the next cycle low, based on:
- Asset-specific cycle lengths
- Current market position
- Historical cycle patterns

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is for informational purposes only. Always conduct your own research and consider consulting with a financial advisor before making investment decisions.
