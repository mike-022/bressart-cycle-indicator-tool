import streamlit as st
import pandas as pd
from data_fetcher import DataFetcher
from bressart_cycle_indicator import BressartCycleIndicator
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def create_plotly_figure(analyzer, start_date=None, end_date=None):
    """Create an interactive Plotly figure for the cycle analysis."""
    if start_date is None:
        start_date = analyzer.data.index[0]
    if end_date is None:
        end_date = analyzer.data.index[-1]
        
    mask = (analyzer.data.index >= start_date) & (analyzer.data.index <= end_date)
    plot_data = analyzer.data[mask]
    
    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.03,
                       row_heights=[0.7, 0.3])

    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=plot_data.index,
        open=plot_data['Open'],
        high=plot_data['High'],
        low=plot_data['Low'],
        close=plot_data['Close'],
        name='Price',
        text=[f"Date: {date}<br>" +
              f"Open: {open:.2f}<br>" +
              f"High: {high:.2f}<br>" +
              f"Low: {low:.2f}<br>" +
              f"Close: {close:.2f}"
              for date, open, high, low, close in zip(plot_data.index,
                                                    plot_data['Open'],
                                                    plot_data['High'],
                                                    plot_data['Low'],
                                                    plot_data['Close'])],
        hoverinfo='text',
        showlegend=False  # Hide from legend
    ), row=1, col=1)

    # Add essential SMAs
    fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['SMA20'],
                            name='20 SMA', line=dict(color='orange', width=1)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['SMA50'],
                            name='50 SMA', line=dict(color='red', width=1)),
                  row=1, col=1)

    # Add volume with cleaner look
    if 'Volume' in plot_data.columns:
        fig.add_trace(go.Bar(x=plot_data.index, y=plot_data['Volume'],
                            name='Volume', marker_color='rgba(128, 128, 128, 0.3)',
                            showlegend=False),  # Hide from legend
                     row=2, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Volume_SMA20'],
                                name='Volume SMA20', line=dict(color='rgba(255, 0, 0, 0.5)', width=1)),
                     row=2, col=1)

    # Analyze cycles
    analysis = analyzer.analyze_cycles()
    
    # Only show cycles based on the interval
    if analyzer.interval == 'daily':
        cycles_to_show = analysis['daily']
        cycle_prefix = 'DC'
    else:
        cycles_to_show = analysis['weekly']
        cycle_prefix = 'WC'
    
    # Plot cycles
    cycle_lows_shown = False
    cycle_highs_shown = False
    
    for cycle in cycles_to_show:
        start = cycle['start']
        end = cycle['end']
        
        # Add cycle low marker
        cycle_low_price = plot_data['Low'].iloc[start]
        cycle_low_date = plot_data.index[start]
        marker_color = 'green' if cycle_prefix == 'DC' else 'purple'
        marker_symbol = 'triangle-up' if cycle_prefix == 'DC' else 'diamond'
        marker_size = 12 if cycle_prefix == 'DC' else 15
        
        fig.add_trace(go.Scatter(x=[cycle_low_date],
                               y=[cycle_low_price],
                               mode='markers+text',
                               marker=dict(color=marker_color, size=marker_size, symbol=marker_symbol),
                               text=[f'{cycle_prefix}L {cycle_low_date.strftime("%m/%d")}'],
                               textposition="bottom center",
                               name=f'{"Daily" if cycle_prefix == "DC" else "Weekly"} Cycle Low',
                               showlegend=not cycle_lows_shown),  # Only show in legend once
                     row=1, col=1)
        cycle_lows_shown = True
        
        # Add cycle high marker
        cycle_slice = slice(start, end + 1)
        cycle_data = plot_data.iloc[cycle_slice]
        high_idx = cycle_data['High'].idxmax()
        high_price = cycle_data['High'].max()
        
        fig.add_trace(go.Scatter(x=[high_idx],
                               y=[high_price],
                               mode='markers+text',
                               marker=dict(color=marker_color, size=marker_size, symbol='triangle-down'),
                               text=[f'{cycle_prefix}H {high_idx.strftime("%m/%d")}'],
                               textposition="top center",
                               name=f'{"Daily" if cycle_prefix == "DC" else "Weekly"} Cycle High',
                               showlegend=not cycle_highs_shown),  # Only show in legend once
                     row=1, col=1)
        cycle_highs_shown = True

    # Add candidate cycle low prediction and timing window
    if cycles_to_show:
        last_cycle = cycles_to_show[-1]
        last_cycle_low_date = plot_data.index[last_cycle['end']]  # Use end of last cycle
        
        # Calculate timing window based on interval
        if analyzer.interval == 'daily':
            min_days = analyzer.cycle_params[analyzer.asset_type]['daily'][0]  # Use asset-specific timing
            max_days = analyzer.cycle_params[analyzer.asset_type]['daily'][1]
        else:  # weekly
            min_days = analyzer.cycle_params[analyzer.asset_type]['weekly'][0] * 7
            max_days = analyzer.cycle_params[analyzer.asset_type]['weekly'][1] * 7
        
        # Calculate window dates
        earliest_low = last_cycle_low_date + pd.Timedelta(days=min_days * 0.6)  # Allow for early cycles
        latest_low = last_cycle_low_date + pd.Timedelta(days=max_days * 1.4)  # Allow for late cycles
        
        # Only show if we're approaching the window
        current_date = plot_data.index[-1]
        if current_date >= earliest_low - pd.Timedelta(days=min_days/2):
            # Add timing window rectangle
            fig.add_shape(
                type="rect",
                x0=earliest_low,
                x1=latest_low,
                y0=plot_data['Low'].min(),
                y1=plot_data['High'].max(),
                fillcolor="rgba(255, 255, 0, 0.1)",
                line=dict(color="yellow", width=1, dash="dash"),
                name="Timing Window",
                row=1, col=1
            )
            
            # Add annotation for timing window
            fig.add_annotation(
                x=earliest_low + (latest_low - earliest_low)/2,
                y=plot_data['High'].max(),
                text=f"Next {cycle_prefix}L Window",
                showarrow=False,
                yshift=20,
                font=dict(color="yellow"),
                row=1, col=1
            )

    # Update layout for cleaner look
    fig.update_layout(
        title=f'{analyzer.asset_type} {analyzer.interval.capitalize()} Cycle Analysis',
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis2_title="Volume",
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)"
        ),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Clean up axis labels and grid
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

def main():
    st.set_page_config(page_title="Bressart Cycle Indicator Tool", layout="wide")
    
    st.title("Bressart Cycle Indicator Tool")
    st.write("Analyze market cycles using cycle theory")
    
    # Initialize data fetcher
    fetcher = DataFetcher()
    
    # Add settings section
    st.sidebar.header("Settings")
    
    # Asset selection
    st.sidebar.subheader("Select Asset")
    default_assets = ["BITCOIN", "ETHEREUM", "SP500", "NASDAQ", "DOW", "GOLD", "OIL", "DXY"]
    use_custom_asset = st.sidebar.checkbox("Use Custom Asset", value=False)
    
    if use_custom_asset:
        asset_type = st.sidebar.text_input("Enter Asset Symbol (e.g., AAPL, MSFT)")
        # Add cycle length inputs for custom assets
        st.sidebar.subheader("Custom Cycle Parameters")
        daily_min = st.sidebar.number_input("Daily Cycle Min (days)", value=20, min_value=5, max_value=100)
        daily_max = st.sidebar.number_input("Daily Cycle Max (days)", value=40, min_value=daily_min, max_value=120)
        weekly_min = st.sidebar.number_input("Weekly Cycle Min (weeks)", value=5, min_value=2, max_value=52)
        weekly_max = st.sidebar.number_input("Weekly Cycle Max (weeks)", value=13, min_value=weekly_min, max_value=52)
        yearly_cycles = st.sidebar.number_input("Yearly Cycles", value=4, min_value=1, max_value=10)
    else:
        asset_type = st.sidebar.selectbox("Select Asset", default_assets)
    
    # Time period selection
    st.sidebar.subheader("Select Time Period")
    period = st.sidebar.selectbox(
        "Select Time Period",
        ["1y", "2y", "5y", "max"]
    )
    
    # Interval selection
    st.sidebar.subheader("Select Interval")
    interval = st.sidebar.selectbox(
        "Select Interval",
        ["daily", "weekly"]
    )
    
    # Add CMA strategy toggle
    st.sidebar.subheader("Cycle Detection Method")
    use_cma = st.sidebar.checkbox("Use Centered MA Strategy", 
                                 value=True if use_custom_asset else False,
                                 help="Toggle between standard cycle detection and centered moving average strategy")
    
    # Date range (optional)
    st.sidebar.subheader("Date Range")
    use_custom_dates = st.sidebar.checkbox("Use Custom Date Range")
    
    start_date = None
    end_date = None
    if use_custom_dates:
        start_date = st.sidebar.date_input("Start Date")
        end_date = st.sidebar.date_input("End Date")
    
    # Fetch and analyze data
    try:
        with st.spinner("Fetching data..."):
            data = fetcher.fetch_data(asset_type, interval, period)
            
        with st.spinner("Analyzing cycles..."):
            # If using custom asset, create custom cycle parameters
            if use_custom_asset:
                custom_params = {
                    asset_type: {
                        'daily': (daily_min, daily_max),
                        'weekly': (weekly_min, weekly_max),
                        'yearly': yearly_cycles
                    }
                }
                analyzer = BressartCycleIndicator(data, asset_type, interval, custom_params=custom_params)
            else:
                analyzer = BressartCycleIndicator(data, asset_type, interval)
            
            analyzer.use_cma_strategy = use_cma
            
            # Create and display the plot
            fig = create_plotly_figure(analyzer, start_date, end_date)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display cycle statistics
            st.header("Cycle Statistics")
            
            analysis = analyzer.analyze_cycles()
            
            # Create two columns for daily and weekly cycles
            daily_col, weekly_col = st.columns(2)
            
            # Daily Cycles
            with daily_col:
                st.subheader("Daily Cycles")
                daily_cycles = analysis['daily']
                if daily_cycles:
                    for i, cycle in enumerate(daily_cycles, 1):
                        cycle_data = analyzer.data.iloc[cycle['start']:cycle['end']+1]
                        cycle_days = (analyzer.data.index[cycle['end']] - analyzer.data.index[cycle['start']]).days
                        cycle_high = cycle_data['High'].max()
                        cycle_low = cycle_data['Low'].min()
                        cycle_return = ((cycle_high - cycle_low) / cycle_low) * 100
                        
                        st.markdown(f"""
                        **Cycle {i}** ({cycle['translation']})
                        - Duration: {cycle_days} days
                        - Low: ${cycle_low:,.2f} ({analyzer.data.index[cycle['start']].strftime('%Y-%m-%d')})
                        - High: ${cycle_high:,.2f} ({cycle_data['High'].idxmax().strftime('%Y-%m-%d')})
                        - Return: {cycle_return:.1f}%
                        - Status: {'❌ Failed' if cycle['failed'] else '✅ Valid'}
                        """)
                else:
                    st.write("No daily cycles detected")
            
            # Weekly Cycles
            with weekly_col:
                st.subheader("Weekly Cycles")
                weekly_cycles = analysis['weekly']
                if weekly_cycles:
                    for i, cycle in enumerate(weekly_cycles, 1):
                        cycle_data = analyzer.data.iloc[cycle['start']:cycle['end']+1]
                        cycle_days = (analyzer.data.index[cycle['end']] - analyzer.data.index[cycle['start']]).days
                        cycle_weeks = cycle_days // 7
                        cycle_high = cycle_data['High'].max()
                        cycle_low = cycle_data['Low'].min()
                        cycle_return = ((cycle_high - cycle_low) / cycle_low) * 100
                        
                        st.markdown(f"""
                        **Cycle {i}** ({cycle['translation']})
                        - Duration: {cycle_weeks} weeks
                        - Low: ${cycle_low:,.2f} ({analyzer.data.index[cycle['start']].strftime('%Y-%m-%d')})
                        - High: ${cycle_high:,.2f} ({cycle_data['High'].idxmax().strftime('%Y-%m-%d')})
                        - Return: {cycle_return:.1f}%
                        - Status: {'❌ Failed' if cycle['failed'] else '✅ Valid'}
                        """)
                else:
                    st.write("No weekly cycles detected")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 