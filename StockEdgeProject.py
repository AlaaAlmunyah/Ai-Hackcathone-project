import streamlit as st
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt


# Streamlit app title
st.title('Stock Price Forecasting with ARIMA and Momentum Indicator Analysis')


# Display image
st.image('StockEdge1.jpeg', caption='Image Caption', use_column_width=True)

# User inputs
ticker = st.text_input('Enter Stock Ticker', 'AAPL')
start_date = st.date_input('Select Start Date', value=pd.to_datetime('2022-01-01'))


# Function to load stock data
@st.cache
def load_data(ticker, start_date):
    stock = yf.Ticker(ticker)
    history = stock.history(start=start_date, interval='1d')
    return history


# Load and display stock data
try:
    hist = load_data(ticker, start_date)
    if not hist.empty:
        st.write(f"Closing Prices for {ticker}:")
        st.line_chart(hist['Close'])

        # ARIMA Model
        time_series = hist['Close']
        model = ARIMA(time_series, order=(5, 1, 0))
        model_fit = model.fit()
        future_price = model_fit.predict(start=len(time_series), end=len(time_series) + 6)

        # Momentum Indicator
        momentum_indicator = future_price.diff().fillna(0)
        min_val = momentum_indicator.min()
        max_val = momentum_indicator.max()
        momentum_indicator_scaled = 100 * (momentum_indicator - min_val) / (max_val - min_val)

        # Prepare DataFrame for results
        future_dates = pd.date_range(start=time_series.index[-1] + pd.Timedelta(days=1), periods=7, freq='B')
        results_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': future_price,
            'Momentum_Indicator': momentum_indicator_scaled
        })


        # Status determination based on Momentum Indicator
        def determine_status(momentum_value):
            if momentum_value > 80:
                return 'Overbought'
            elif momentum_value < 20:
                return 'Oversold'
            else:
                return 'Neutral'


        results_df['Status'] = results_df['Momentum_Indicator'].apply(determine_status)

        # Display results
        st.write("Predicted Closing Prices and Momentum Indicator:")
        st.dataframe(results_df.set_index('Date'))

        # Plotting Predictions and Momentum Indicator
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Predicted Closing Prices
        axs[0].plot(results_df['Date'], results_df['Predicted_Close'], marker='o', color='blue',
                    label='Predicted Close')
        axs[0].set_title(f'Predicted Closing Prices for {ticker}')
        axs[0].set_ylabel('Price')
        axs[0].legend()

        # Momentum Indicator
        axs[1].plot(results_df['Date'], results_df['Momentum_Indicator'], marker='x', color='blue',
                    label='Momentum Indicator')
        axs[1].set_title('Momentum Indicator')
        axs[1].set_ylabel('Indicator Value')
        axs[1].axhline(80, color='red', linestyle='--', label='Overbought Line')
        axs[1].axhline(20, color='blue', linestyle='--', label='Oversold Line')
        axs[1].legend()

        st.pyplot(fig)
    else:
        st.error("No data available for the selected ticker and start date.")
except Exception as e:
    st.error(f"An error occurred: {e}")
