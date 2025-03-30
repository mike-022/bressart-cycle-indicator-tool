import streamlit as st
import pandas as pd
from data_fetcher import DataFetcher
from bressart_cycle_indicator import BressartCycleIndicator
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def create_plotly_figure(analyzer, start_date=None, end_date=None, visualization_options=None):
    """Create an interactive Plotly figure for the cycle analysis."""
    if visualization_options is None:
        visualization_options = {
            'show_bollinger': False,
            'show_stochastic': False,
            'show_macd': False,
            'show_volume': True,
            'show_confirmation_scores': True
        }
    
    if start_date is None:
        start_date = analyzer.data.index[0]
    if end_date is None:
        end_date = analyzer.data.index[-1]
        
    mask = (analyzer.data.index >= start_date) & (analyzer.data.index <= end_date)
    plot_data = analyzer.data[mask]
    
    # Create figure with secondary y-axis
    fig = make_subplots(rows=5, cols=1,  # Added one more row for RSI3M3
                       shared_xaxes=True,
                       vertical_spacing=0.02,
                       row_heights=[0.4, 0.15, 0.15, 0.15, 0.15])  # Adjusted row heights

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
        showlegend=False
    ), row=1, col=1)

    # Add Bollinger Bands if selected
    if visualization_options['show_bollinger']:
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['BB_Upper'],
                                name='BB Upper', line=dict(color='gray', width=1, dash='dash')),
                     row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['BB_Lower'],
                                name='BB Lower', line=dict(color='gray', width=1, dash='dash')),
                     row=1, col=1)

    # Add moving averages
    fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['SMA20'],
                            name='20 SMA', line=dict(color='orange', width=1)),
                 row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['SMA50'],
                            name='50 SMA', line=dict(color='red', width=1)),
                 row=1, col=1)

    # Add Stochastic if selected
    if visualization_options['show_stochastic']:
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Stoch_K'],
                                name='Stoch %K', line=dict(color='blue', width=1)),
                     row=2, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Stoch_D'],
                                name='Stoch %D', line=dict(color='red', width=1)),
                     row=2, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="gray", row=2, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="gray", row=2, col=1)

    # Add MACD if selected
    if visualization_options['show_macd']:
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MACD'],
                                name='MACD', line=dict(color='blue', width=1)),
                     row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MACD_Signal'],
                                name='Signal', line=dict(color='orange', width=1)),
                     row=3, col=1)
        fig.add_trace(go.Bar(x=plot_data.index, y=plot_data['MACD_Hist'],
                            name='MACD Hist', marker_color='gray'),
                     row=3, col=1)

    # Add volume if selected
    if visualization_options['show_volume'] and 'Volume' in plot_data.columns:
        fig.add_trace(go.Bar(x=plot_data.index, y=plot_data['Volume'],
                            name='Volume', marker_color='rgba(128, 128, 128, 0.3)'),
                     row=4, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Volume_SMA20'],
                                name='Vol SMA20', line=dict(color='red', width=1)),
                     row=4, col=1)

    # Add RSI3M3
    rsi3m3 = analyzer.calculate_rsi3m3(analyzer.data['Close'])
    fig.add_trace(go.Scatter(x=plot_data.index, y=rsi3m3[mask],
                            name='RSI3M3', line=dict(color='purple', width=1)),
                 row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    
    # Add Centered Detrend if using that strategy
    if analyzer.use_cma_strategy:
        centered_ma = analyzer.calculate_centered_ma(analyzer.data['Close'])
        detrend = (analyzer.data['Close'] - centered_ma) / centered_ma * 100
        fig.add_trace(go.Scatter(x=plot_data.index, y=detrend[mask],
                                name='Centered Detrend', line=dict(color='blue', width=1)),
                     row=3, col=1)
        fig.add_hline(y=-0.8, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=0.8, line_dash="dash", line_color="red", row=3, col=1)

    # Analyze cycles
    analysis = analyzer.analyze_cycles()
    
    # Only show cycles based on the interval
    if analyzer.interval == 'daily':
        cycles_to_show = analysis['daily']
        cycle_prefix = 'DC'
    else:
        cycles_to_show = analysis['weekly']
        cycle_prefix = 'WC'
    
    # Plot cycles with confirmation scores
    cycle_lows_shown = False
    cycle_highs_shown = False
    
    for cycle in cycles_to_show:
        start = cycle['start']
        end = cycle['end']
        
        # Add cycle low marker with confirmation score if available
        cycle_low_price = plot_data['Low'].iloc[start]
        cycle_low_date = plot_data.index[start]
        marker_color = 'green' if cycle_prefix == 'DC' else 'purple'
        marker_symbol = 'triangle-up' if cycle_prefix == 'DC' else 'diamond'
        marker_size = 12 if cycle_prefix == 'DC' else 15
        
        # Create hover text with confirmation signals if available
        hover_text = f'{cycle_prefix}L {cycle_low_date.strftime("%m/%d")}'
        if 'confirmation_signals' in cycle and visualization_options['show_confirmation_scores']:
            signals = cycle['confirmation_signals']
            hover_text += f"<br>Score: {signals['confirmation_score']}%"
            hover_text += f"<br>RSI: {signals['momentum']['rsi']:.1f}"
            hover_text += f"<br>Stoch: {signals['momentum']['stoch_k']:.1f}"
            if 'volume' in signals:
                hover_text += f"<br>Volume: {signals['volume']['volume_spike']}"
        
        fig.add_trace(go.Scatter(x=[cycle_low_date],
                               y=[cycle_low_price],
                               mode='markers+text',
                               marker=dict(color=marker_color, 
                                         size=marker_size, 
                                         symbol=marker_symbol),
                               text=[hover_text],
                               textposition="bottom center",
                               name=f'{"Daily" if cycle_prefix == "DC" else "Weekly"} Cycle Low',
                               showlegend=not cycle_lows_shown),
                     row=1, col=1)
        cycle_lows_shown = True
        
        # Add entry point and stop loss for RSI3M3 strategy
        if not analyzer.use_cma_strategy and 'setup_bar' in cycle:
            setup_date = cycle['signal_date']
            entry_price = cycle['entry_price']
            stop_price = cycle['stop_price']
            
            # Add entry point
            fig.add_trace(go.Scatter(x=[setup_date],
                                   y=[entry_price],
                                   mode='markers+text',
                                   marker=dict(color='yellow', size=10, symbol='circle'),
                                   text=['Entry'],
                                   textposition="top center",
                                   name='Entry Point',
                                   showlegend=False),
                         row=1, col=1)
            
            # Add stop loss
            fig.add_trace(go.Scatter(x=[setup_date],
                                   y=[stop_price],
                                   mode='markers+text',
                                   marker=dict(color='red', size=10, symbol='circle'),
                                   text=['Stop'],
                                   textposition="bottom center",
                                   name='Stop Loss',
                                   showlegend=False),
                         row=1, col=1)
        
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
        # Find the last cycle low
        last_low = None
        for cycle in cycles_to_show:
            if cycle['start'] < len(plot_data):
                last_low = cycle['start']
        
        if last_low is not None:
            last_cycle_low_date = plot_data.index[last_low]
            
            # Get timing windows using the new function
            extended_earliest, extended_latest, normal_earliest, normal_latest = analyzer.calculate_next_low_window(
                last_cycle_low_date,
                'daily' if analyzer.interval == 'daily' else 'weekly'
            )
            
            # Calculate y-range for the windows
            y_range = plot_data['High'].max() - plot_data['Low'].min()
            y_min = plot_data['Low'].min() - y_range * 0.05  # Add 5% padding
            y_max = plot_data['High'].max() + y_range * 0.05
            
            # Add extended timing window (lighter shade)
            fig.add_vrect(
                x0=extended_earliest,
                x1=extended_latest,
                fillcolor="rgba(255, 235, 235, 0.1)",  # Very light red
                line=dict(color="red", width=1, dash="dash"),
                layer="below",
                name="Extended Window",
                row=1, col=1
            )
            
            # Add normal timing window (darker shade)
            fig.add_vrect(
                x0=normal_earliest,
                x1=normal_latest,
                fillcolor="rgba(255, 235, 235, 0.2)",  # Light red
                line=dict(color="red", width=1),
                layer="below",
                name="Normal Window",
                row=1, col=1
            )
            
            # Add window labels with better positioning
            y_label_pos = y_min
            
            # Normal window labels
            fig.add_annotation(
                x=normal_earliest,
                y=y_label_pos,
                text=f"Normal Window<br>{normal_earliest.strftime('%Y-%m-%d')}",
                showarrow=True,
                arrowhead=1,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red",
                ax=0,
                ay=40,
                font=dict(size=10, color="red"),
                row=1, col=1
            )
            
            fig.add_annotation(
                x=normal_latest,
                y=y_label_pos,
                text=f"Normal End<br>{normal_latest.strftime('%Y-%m-%d')}",
                showarrow=True,
                arrowhead=1,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red",
                ax=0,
                ay=40,
                font=dict(size=10, color="red"),
                row=1, col=1
            )
            
            # Extended window labels
            fig.add_annotation(
                x=extended_earliest,
                y=y_label_pos,
                text=f"Extended Start<br>{extended_earliest.strftime('%Y-%m-%d')}",
                showarrow=True,
                arrowhead=1,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="rgba(255, 0, 0, 0.5)",
                ax=0,
                ay=60,
                font=dict(size=8, color="rgba(255, 0, 0, 0.5)"),
                row=1, col=1
            )
            
            fig.add_annotation(
                x=extended_latest,
                y=y_label_pos,
                text=f"Extended End<br>{extended_latest.strftime('%Y-%m-%d')}",
                showarrow=True,
                arrowhead=1,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="rgba(255, 0, 0, 0.5)",
                ax=0,
                ay=60,
                font=dict(size=8, color="rgba(255, 0, 0, 0.5)"),
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
    
    # Add sidebar controls
    with st.sidebar:
        st.header("Settings")
        
        # Asset selection
        st.subheader("Select Asset")
        use_custom_asset = st.checkbox("Use Custom Asset")
        
        if use_custom_asset:
            asset = st.text_input("Enter Asset Name").upper()
            custom_params = {
                asset: {
                    'daily': (st.number_input("Min Daily Cycle Length", value=20), 
                            st.number_input("Max Daily Cycle Length", value=30)),
                    'weekly': (st.number_input("Min Weekly Cycle Length", value=20), 
                             st.number_input("Max Weekly Cycle Length", value=30)),
                    'yearly': st.number_input("Yearly Cycle Count", value=4)
                }
            }
        else:
            asset = st.selectbox("Select Asset", 
                               ["BITCOIN", "ETHEREUM", "SP500", "NASDAQ", "DOW", "GOLD", "OIL", "DXY"])
            custom_params = None
        
        # Time period selection
        st.subheader("Select Time Period")
        timeframe = st.selectbox("Select Time Period", 
                               ["1y", "2y", "5y", "max"],
                               help="Select the time period for analysis")
        
        # Interval selection
        st.subheader("Select Interval")
        interval = st.selectbox("Select Interval", ["daily", "weekly"])
        
        # Visualization options
        st.subheader("Visualization Options")
        show_bollinger = st.checkbox("Show Bollinger Bands", 
            help='Display Bollinger Bands on the price chart')
        show_stochastic = st.checkbox('Show Stochastic',
            help='Display Stochastic Oscillator')
        show_macd = st.checkbox('Show MACD',
            help='Display MACD indicator')
        show_volume = st.checkbox('Show Volume', value=True,
            help='Display volume analysis')
        show_confirmation_scores = st.checkbox('Show Confirmation Scores', value=True,
            help='Display technical confirmation scores for cycle lows')

        # Debug Information
        st.subheader('Debug Information')
        show_debug = st.checkbox('Show Debug Information',
            help='Display detailed information about potential cycle lows')
        if show_debug:
            debug_days = st.slider('Days to Analyze', 1, 90, 30,
                help='Number of recent days to analyze for potential cycle lows')
        
        # Date range selection
        st.subheader("Date Range")
        use_custom_dates = st.checkbox("Use Custom Date Range")
        
        start_date = None
        end_date = None
        if use_custom_dates:
            start_date = st.date_input("Start Date")
            end_date = st.date_input("End Date")
    
    # Fetch and analyze data
    try:
        with st.spinner("Fetching data..."):
            data = fetcher.fetch_data(asset, interval, timeframe)
            
        with st.spinner("Analyzing cycles..."):
            # Initialize analyzer with selected strategy
            analyzer = BressartCycleIndicator(data, asset, interval=interval, custom_params=custom_params)
            analyzer.use_cma_strategy = (interval == "daily")  # Assuming daily strategy for daily interval
            
            # Create visualization options dictionary
            visualization_options = {
                'show_bollinger': show_bollinger,
                'show_stochastic': show_stochastic,
                'show_macd': show_macd,
                'show_volume': show_volume,
                'show_confirmation_scores': show_confirmation_scores
            }
            
            # Create and display the plot with visualization options
            fig = create_plotly_figure(analyzer, start_date, end_date, visualization_options)
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
                        - Signal Triggered: {cycle['signal_date'].strftime('%Y-%m-%d %H:%M') if cycle['signal_date'] else 'Not found'}
                        - RSI at Low: {analyzer.data['RSI'].iloc[cycle['start']]:.1f}
                        - Price vs MA: {'Below' if analyzer.data['Close'].iloc[cycle['start']] <= analyzer.data['SMA20'].iloc[cycle['start']] else 'Above'} 20 SMA
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
                        - Signal Triggered: {cycle['signal_date'].strftime('%Y-%m-%d %H:%M') if cycle['signal_date'] else 'Not found'}
                        - RSI at Low: {analyzer.data['RSI'].iloc[cycle['start']]:.1f}
                        - Price vs MA: {'Below' if analyzer.data['Close'].iloc[cycle['start']] <= analyzer.data['SMA50'].iloc[cycle['start']] else 'Above'} 50 SMA
                        """)
                else:
                    st.write("No weekly cycles detected")
            
            # Plot cycle lows with explanations and timing windows
            if show_debug:
                cycle_lows = analyzer.detect_cycle_lows()
                
                # Print debug info for potential lows
                end_date = analyzer.data.index[-1]
                start_date = end_date - pd.Timedelta(days=debug_days)
                st.subheader("Debug Information for Recent Potential Cycle Lows")
                st.text("-" * 80)
                analyzer.print_debug_info(start_date, end_date)
                
                # Plot cycle lows and timing windows
                for i, low in enumerate(cycle_lows):
                    # Plot the cycle low marker
                    cycle_low_price = analyzer.data['Low'].iloc[low['index']]
                    cycle_low_date = analyzer.data.index[low['index']]
                    
                    # Create hover text with confirmation signals
                    hover_text = f"Cycle Low {cycle_low_date.strftime('%m/%d')}\n"
                    hover_text += f"Score: {low['confirmation_score']}%\n"
                    hover_text += low['explanation']
                    
                    fig.add_trace(go.Scatter(
                        x=[cycle_low_date],
                        y=[cycle_low_price],
                        mode='markers+text',
                        marker=dict(color='green', size=12, symbol='triangle-up'),
                        text=[hover_text],
                        textposition="bottom center",
                        name='Cycle Low',
                        showlegend=(i == 0)
                    ), row=1, col=1)
                    
                    # Add timing window for next cycle low
                    if i == len(cycle_lows) - 1:  # Only for the last cycle low
                        extended_earliest, extended_latest, normal_earliest, normal_latest = (
                            analyzer.calculate_next_low_window(cycle_low_date)
                        )
                        
                        # Add extended window (lighter shade)
                        fig.add_vrect(
                            x0=extended_earliest,
                            x1=extended_latest,
                            fillcolor="rgba(255, 235, 235, 0.1)",
                            line=dict(color="red", width=1, dash="dash"),
                            layer="below",
                            name="Extended Window"
                        )
                        
                        # Add normal window (darker shade)
                        fig.add_vrect(
                            x0=normal_earliest,
                            x1=normal_latest,
                            fillcolor="rgba(255, 235, 235, 0.2)",
                            line=dict(color="red", width=1),
                            layer="below",
                            name="Normal Window"
                        )
                        
                        # Add window labels
                        y_pos = analyzer.data['Low'].min()
                        fig.add_annotation(
                            x=normal_earliest,
                            y=y_pos,
                            text=f"Normal Window Start\n{normal_earliest.strftime('%Y-%m-%d')}",
                            showarrow=True,
                            arrowhead=1,
                            ax=0,
                            ay=40
                        )
                        
                        fig.add_annotation(
                            x=normal_latest,
                            y=y_pos,
                            text=f"Normal Window End\n{normal_latest.strftime('%Y-%m-%d')}",
                            showarrow=True,
                            arrowhead=1,
                            ax=0,
                            ay=40
                        )

    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 