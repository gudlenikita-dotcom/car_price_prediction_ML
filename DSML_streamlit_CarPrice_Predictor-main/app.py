import pandas as pd
import streamlit as st
import datetime
import numpy as np
import pickle


# Load data
cars_df = pd.read_csv("./cars24-car-price.csv")

st.write(""" 
# 🚘 USED CAR WISPERER""")

# Encoding dictionary for categorical variables
encode_dict = {
    "fuel_type": {"Diesel": 1, "Petrol": 2, "CNG": 3, "LPG": 4, "Electric": 5},
    "transmission_type": {"Manual": 1, "Automatic": 2}
}


def model_pred(fuel_type, transmission_type, engine, seats):
    """Load trained model and predict car price."""
    try:
        with open("car_pred", "rb") as file:
            reg_model = pickle.load(file)

        # Input preparation
        input_features = [[2018.0, 1, 40000, fuel_type, transmission_type, 19.70, engine, 86.30, seats]]
        # Model prediction
        return reg_model.predict(input_features)

    except Exception as e:
        st.error(f"Error in model prediction: {e}")
        return None


# Sidebar for user input
st.sidebar.header("Customize Inputs")
fuel_type = st.sidebar.selectbox("Select Fuel Type", ["Diesel", "Petrol", "CNG", "LPG", "Electric"], help="Choose a fuel type for your car.")
engine = st.sidebar.slider("Engine Power (in cc)", min_value=500, max_value=5000, step=100, help="Select engine power in cubic centimeters.")
transmission_type = st.sidebar.selectbox("Select Transmission Type", ["Manual", "Automatic"], help="Choose the type of transmission.")
seats = st.sidebar.selectbox("Number of Seats", list(range(2, 15)), help="Select the number of seats in your car.")
car_model = st.sidebar.text_input("Enter Car Model (optional)", help="Type the model name if known.")
car_year = st.sidebar.slider("Select Manufacturing Year Range", min_value=1991, max_value=2024, value=(1991, 2024), help="Choose the manufacturing year range.")

# Predict Button
if st.button("Predict Price"):
    # Encode user inputs
    try:
        encoded_fuel_type = encode_dict["fuel_type"][fuel_type]
        encoded_transmission_type = encode_dict["transmission_type"][transmission_type]

        # Perform prediction
        predicted_price = model_pred(encoded_fuel_type, encoded_transmission_type, engine, seats)

        if predicted_price is not None:
            st.success(f"✅ Predicted price: **₹{np.round(predicted_price[0], 2)} Lakhs**")

            # Compare predicted vs historical average
            filtered_comparison_df = cars_df[
                (cars_df["fuel_type"] == fuel_type) &
                (cars_df["transmission_type"] == transmission_type) &
                (cars_df["seats"] == seats) &
                (cars_df["year"].between(car_year[0], car_year[1]))
            ]
            if not filtered_comparison_df.empty:
                historical_mean_price = filtered_comparison_df["selling_price"].mean()  # Use your actual column name
                st.write(
                    f"📊 **Historical Average Price for Similar Cars in Range:** ₹{np.round(historical_mean_price, 2)} Lakhs"
                )
            else:
                st.warning("No historical data found for comparison based on your filters.")
        else:
            st.error("Unable to make prediction due to an issue with the model or inputs.")
    except Exception as e:
        st.error(f"Prediction error: {e}")


# Dataset Exploration
st.write("### 🔍 Explore Dataset")
st.write(
    "You can explore sample data to compare against your prediction inputs."
)

# Filter options
filter_fuel = st.multiselect(
    "Filter by Fuel Type", options=cars_df["fuel_type"].unique(), default=cars_df["fuel_type"].unique()
)

filter_transmission = st.multiselect(
    "Filter by Transmission Type", options=cars_df["transmission_type"].unique(), default=cars_df["transmission_type"].unique()
)

filter_seats = st.slider(
    "Select Number of Seats Range", min_value=int(cars_df["seats"].min()), max_value=int(cars_df["seats"].max()), value=(4, 7)
)

filter_year = st.slider(
    "Select Year Range", min_value=1991, max_value=2024, value=(1991, 2024)
)

filtered_df = cars_df[
    (cars_df["fuel_type"].isin(filter_fuel)) &
    (cars_df["transmission_type"].isin(filter_transmission)) &
    (cars_df["seats"].between(filter_seats[0], filter_seats[1])) &
    (cars_df["year"].between(filter_year[0], filter_year[1]))
]

# Display filtered data
st.write("Filtered Dataset Insights:")
st.dataframe(filtered_df)
