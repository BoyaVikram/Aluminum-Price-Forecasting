import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta

st.title("Aluminium Price Forecasting App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview:")
    st.write(df.head())

    # Ensure Date column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

    # Extract features from date
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    # Store the last available date
    last_date = df['Date'].iloc[-1]

    # Prepare features and target
    X = df[['Day', 'Month', 'Year']]
    y = df['Price']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate model performance
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    st.write("### Model Performance:")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("MSE", f"{mse:.2f}")
    col3.metric("RMSE", f"{rmse:.2f}")
    col4.metric("MAPE", f"{mape:.2f}%")

    # Generate next 365 days
    future_dates = [last_date + timedelta(days=i) for i in range(1, 366)]
    
    # Create dataframe for future dates
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Day': [date.day for date in future_dates],
        'Month': [date.month for date in future_dates],
        'Year': [date.year for date in future_dates]
    })

    # Predict future prices
    future_df['Predicted Price'] = model.predict(future_df[['Day', 'Month', 'Year']])

    # Combine historical & future data
    df_plot = pd.concat([df[['Date', 'Price']], future_df[['Date', 'Predicted Price']]], ignore_index=True)

    # First Visualization
    st.write("### Aluminium Price Trends")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test.values, label='Actual Prices', color='blue')
    ax.plot(y_pred, label='Predicted Prices', color='red', linestyle='dashed')
    ax.axvline(x=len(y_test), color='black', linestyle='dotted', label="Prediction Start")
    ax.plot(range(len(y_test), len(y_test) + 365), future_df['Predicted Price'], label="Next 365 Days", color='green')
    ax.set_xlabel("Samples")
    ax.set_ylabel("Price")
    ax.set_title("Actual vs Predicted Aluminium Prices with 365-Day Forecast")
    ax.legend()
    st.pyplot(fig)

    # Second Visualization with Future Price Hover Effect
    st.write("### Interactive Future Price Chart")
    fig2 = px.line(future_df, x='Date', y='Predicted Price', title='Future Price Trend', markers=True, hover_data={'Date': True, 'Predicted Price': True})
    fig2.update_traces(mode='markers+lines', marker=dict(size=6))
    st.plotly_chart(fig2, use_container_width=True)

    # Show forecasted data
    st.write("### Forecasted Prices for Next 365 Days:")
    st.write(future_df)
