import tkinter as tk
from tkinter import messagebox
import requests
import pandas as pd
import math
from datetime import datetime

def download_data():
    tickers = ticker_entry.get().strip()
    api_key = api_key_entry.get().strip()  # Used for Alpha Vantage; optional for CryptoCompare
    start_date = start_date_entry.get().strip()  # Optional
    end_date = end_date_entry.get().strip()      # Optional
    asset_type = asset_type_var.get()            # "Stock" or "Crypto"
    
    if not tickers:
        messagebox.showerror("Input Error", "Please enter at least one ticker symbol.")
        return
    if asset_type == "Stock" and not api_key:
        messagebox.showerror("Input Error", "Please enter your Alpha Vantage API key for stocks.")
        return

    tickers_list = [ticker.strip() for ticker in tickers.split(",") if ticker.strip()]
    
    # Process each ticker.
    for ticker in tickers_list:
        if asset_type == "Crypto":
            # For Crypto, use CryptoCompare endpoints.
            # Set end timestamp (if not provided, use current time)
            if end_date:
                try:
                    end_dt = pd.to_datetime(end_date)
                    end_ts = int(end_dt.timestamp())
                except Exception as e:
                    messagebox.showerror("Date Error", f"Error parsing end date: {end_date}\n{e}")
                    return
            else:
                end_dt = pd.to_datetime(datetime.now())
                end_ts = int(datetime.now().timestamp())
            
            # --- Daily Crypto Data ---
            if daily_var.get():
                if start_date:
                    try:
                        start_dt = pd.to_datetime(start_date)
                        total_days = (end_dt - start_dt).days
                        limit = total_days
                        if limit > 2000:
                            limit = 2000
                    except Exception as e:
                        messagebox.showerror("Date Error", f"Error computing days from start date: {start_date}\n{e}")
                        return
                else:
                    limit = 2000
                url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={ticker.upper()}&tsym=USD&aggregate=1&limit={limit}&toTs={end_ts}"
                headers = {}
                if api_key:
                    headers = {"authorization": f"Apikey {api_key}"}
                try:
                    response = requests.get(url, headers=headers)
                    data = response.json()
                    if data.get("Response") != "Success":
                        msg = data.get("Message", "Unknown error")
                        messagebox.showerror("API Error", f"Could not retrieve daily crypto data for {ticker}. {msg}")
                    else:
                        raw_data = data["Data"]["Data"]
                        if not raw_data:
                            messagebox.showwarning("Data Warning", f"No daily data returned for {ticker}")
                        else:
                            df = pd.DataFrame(raw_data)
                            df["time"] = pd.to_datetime(df["time"], unit="s")
                            df.set_index("time", inplace=True)
                            df.sort_index(inplace=True)
                            if start_date:
                                df = df[df.index >= pd.to_datetime(start_date)]
                            filename = f"{ticker}_daily_crypto.csv"
                            df.to_csv(filename)
                except Exception as e:
                    messagebox.showerror("Download Error", f"Error downloading daily crypto data for {ticker}:\n{e}")
            
            # --- Weekly Crypto Data ---
            if weekly_var.get():
                if start_date:
                    try:
                        start_dt = pd.to_datetime(start_date)
                        total_weeks = math.ceil((end_dt - start_dt).days / 7)
                        limit = total_weeks - 1  # CryptoCompare returns limit+1 points
                        if limit > 2000:
                            limit = 2000
                    except Exception as e:
                        messagebox.showerror("Date Error", f"Error computing weeks from start date: {start_date}\n{e}")
                        return
                else:
                    limit = 2000
                # Use histoday with aggregate=7 for weekly data.
                url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={ticker.upper()}&tsym=USD&aggregate=7&limit={limit}&toTs={end_ts}"
                headers = {}
                if api_key:
                    headers = {"authorization": f"Apikey {api_key}"}
                try:
                    response = requests.get(url, headers=headers)
                    data = response.json()
                    if data.get("Response") != "Success":
                        msg = data.get("Message", "Unknown error")
                        messagebox.showerror("API Error", f"Could not retrieve weekly crypto data for {ticker}. {msg}")
                    else:
                        raw_data = data["Data"]["Data"]
                        if not raw_data:
                            messagebox.showwarning("Data Warning", f"No weekly data returned for {ticker}")
                        else:
                            df = pd.DataFrame(raw_data)
                            df["time"] = pd.to_datetime(df["time"], unit="s")
                            df.set_index("time", inplace=True)
                            df.sort_index(inplace=True)
                            if start_date:
                                df = df[df.index >= pd.to_datetime(start_date)]
                            filename = f"{ticker}_weekly_crypto.csv"
                            df.to_csv(filename)
                except Exception as e:
                    messagebox.showerror("Download Error", f"Error downloading weekly crypto data for {ticker}:\n{e}")
            
            # --- Monthly Crypto Data ---
            if monthly_var.get():
                if start_date:
                    try:
                        start_dt = pd.to_datetime(start_date)
                        total_months = math.ceil((end_dt - start_dt).days / 30)  # approximate months
                        limit = total_months - 1
                        if limit > 2000:
                            limit = 2000
                    except Exception as e:
                        messagebox.showerror("Date Error", f"Error computing months from start date: {start_date}\n{e}")
                        return
                else:
                    limit = 2000
                # Use the histomonth endpoint for monthly data.
                url = f"https://min-api.cryptocompare.com/data/v2/histomonth?fsym={ticker.upper()}&tsym=USD&limit={limit}&toTs={end_ts}"
                headers = {}
                if api_key:
                    headers = {"authorization": f"Apikey {api_key}"}
                try:
                    response = requests.get(url, headers=headers)
                    data = response.json()
                    if data.get("Response") != "Success":
                        msg = data.get("Message", "Unknown error")
                        messagebox.showerror("API Error", f"Could not retrieve monthly crypto data for {ticker}. {msg}")
                    else:
                        raw_data = data["Data"]["Data"]
                        if not raw_data:
                            messagebox.showwarning("Data Warning", f"No monthly data returned for {ticker}")
                        else:
                            df = pd.DataFrame(raw_data)
                            df["time"] = pd.to_datetime(df["time"], unit="s")
                            df.set_index("time", inplace=True)
                            df.sort_index(inplace=True)
                            if start_date:
                                df = df[df.index >= pd.to_datetime(start_date)]
                            filename = f"{ticker}_monthly_crypto.csv"
                            df.to_csv(filename)
                except Exception as e:
                    messagebox.showerror("Download Error", f"Error downloading monthly crypto data for {ticker}:\n{e}")
        else:
            # For Stocks, use Alpha Vantage endpoints.
            # --- Daily Stock Data ---
            if daily_var.get():
                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}"
                try:
                    response = requests.get(url)
                    data = response.json()
                    key = "Time Series (Daily)"
                    if key not in data:
                        messagebox.showerror("API Error", f"Could not retrieve daily stock data for {ticker}.")
                    else:
                        daily_data = data[key]
                        df = pd.DataFrame.from_dict(daily_data, orient='index')
                        df.index = pd.to_datetime(df.index)
                        df.sort_index(inplace=True)
                        df.rename(columns={
                            "1. open": "Open",
                            "2. high": "High",
                            "3. low": "Low",
                            "4. close": "Close",
                            "5. volume": "Volume"
                        }, inplace=True)
                        if start_date:
                            df = df[df.index >= pd.to_datetime(start_date)]
                        if end_date:
                            df = df[df.index <= pd.to_datetime(end_date)]
                        filename = f"{ticker}_daily_stock.csv"
                        df.to_csv(filename)
                except Exception as e:
                    messagebox.showerror("Download Error", f"Error downloading daily stock data for {ticker}:\n{e}")
            # --- Weekly Stock Data ---
            if weekly_var.get():
                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={ticker}&apikey={api_key}"
                try:
                    response = requests.get(url)
                    data = response.json()
                    key = "Weekly Time Series"
                    if key not in data:
                        messagebox.showerror("API Error", f"Could not retrieve weekly stock data for {ticker}.")
                    else:
                        weekly_data = data[key]
                        df = pd.DataFrame.from_dict(weekly_data, orient='index')
                        df.index = pd.to_datetime(df.index)
                        df.sort_index(inplace=True)
                        df.rename(columns={
                            "1. open": "Open",
                            "2. high": "High",
                            "3. low": "Low",
                            "4. close": "Close",
                            "5. volume": "Volume"
                        }, inplace=True)
                        if start_date:
                            df = df[df.index >= pd.to_datetime(start_date)]
                        if end_date:
                            df = df[df.index <= pd.to_datetime(end_date)]
                        filename = f"{ticker}_weekly_stock.csv"
                        df.to_csv(filename)
                except Exception as e:
                    messagebox.showerror("Download Error", f"Error downloading weekly stock data for {ticker}:\n{e}")
            # --- Monthly Stock Data ---
            if monthly_var.get():
                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol={ticker}&apikey={api_key}"
                try:
                    response = requests.get(url)
                    data = response.json()
                    key = "Monthly Time Series"
                    if key not in data:
                        messagebox.showerror("API Error", f"Could not retrieve monthly stock data for {ticker}.")
                    else:
                        monthly_data = data[key]
                        df = pd.DataFrame.from_dict(monthly_data, orient='index')
                        df.index = pd.to_datetime(df.index)
                        df.sort_index(inplace=True)
                        df.rename(columns={
                            "1. open": "Open",
                            "2. high": "High",
                            "3. low": "Low",
                            "4. close": "Close",
                            "5. volume": "Volume"
                        }, inplace=True)
                        if start_date:
                            df = df[df.index >= pd.to_datetime(start_date)]
                        if end_date:
                            df = df[df.index <= pd.to_datetime(end_date)]
                        filename = f"{ticker}_monthly_stock.csv"
                        df.to_csv(filename)
                except Exception as e:
                    messagebox.showerror("Download Error", f"Error downloading monthly stock data for {ticker}:\n{e}")
    
    messagebox.showinfo("Success", "Data download complete. Check the CSV files in your directory.")

# ----- UI Setup -----
root = tk.Tk()
root.title("OHLC Data Downloader")

# Ticker symbols input.
tk.Label(root, text="Ticker Symbols (comma separated):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
ticker_entry = tk.Entry(root, width=50)
ticker_entry.grid(row=0, column=1, padx=5, pady=5)

# API Key input.
tk.Label(root, text="API Key (Alpha Vantage / optionally CryptoCompare):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
api_key_entry = tk.Entry(root, width=50)
api_key_entry.grid(row=1, column=1, padx=5, pady=5)

# Start date input (optional).
tk.Label(root, text="Start Date (YYYY-MM-DD, optional):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
start_date_entry = tk.Entry(root, width=20)
start_date_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

# End date input (optional).
tk.Label(root, text="End Date (YYYY-MM-DD, optional):").grid(row=3, column=0, padx=5, pady=5, sticky="w")
end_date_entry = tk.Entry(root, width=20)
end_date_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")

# Asset type selection.
tk.Label(root, text="Asset Type:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
asset_type_var = tk.StringVar(root)
asset_type_var.set("Stock")  # Default selection.
asset_type_menu = tk.OptionMenu(root, asset_type_var, "Stock", "Crypto")
asset_type_menu.grid(row=4, column=1, padx=5, pady=5, sticky="w")

# Timeframe selection checkboxes.
tk.Label(root, text="Select Timeframes:").grid(row=5, column=0, padx=5, pady=5, sticky="w")
frame = tk.Frame(root)
frame.grid(row=5, column=1, padx=5, pady=5, sticky="w")
daily_var = tk.BooleanVar(value=False)
weekly_var = tk.BooleanVar(value=True)
monthly_var = tk.BooleanVar(value=False)
tk.Checkbutton(frame, text="Daily", variable=daily_var).grid(row=0, column=0, sticky="w")
tk.Checkbutton(frame, text="Weekly", variable=weekly_var).grid(row=0, column=1, sticky="w")
tk.Checkbutton(frame, text="Monthly", variable=monthly_var).grid(row=0, column=2, sticky="w")

# Download button.
download_button = tk.Button(root, text="Download Data", command=download_data)
download_button.grid(row=6, column=0, columnspan=2, pady=10)

root.mainloop()
