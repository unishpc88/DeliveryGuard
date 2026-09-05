import streamlit as st
import joblib
import math
import pandas as pd

st.title("DeliveryGuard")
st.write("Food Delivery Time Prediction System")

model_path = "models/deliveryguard_model.pkl"
model = joblib.load(model_path)

st.success("Model loaded successfully!")

st.header("Delivery Prediction")

delivery_age = st.number_input(
    "Delivery Person Age",
    min_value=15,
    max_value=50,
    value=30
)

delivery_rating = st.number_input(
    "Delivery Person Rating",
    min_value=1.0,
    max_value=6.0,
    value=4.7,
    step=0.1
)

weather = st.selectbox(
    "Weather Condition",
    [
        "conditions Sunny",
        "conditions Stormy",
        "conditions Sandstorms",
        "conditions Cloudy",
        "conditions Fog",
        "conditions Windy"
    ]
)

traffic = st.selectbox(
    "Road Traffic Density",
    [
        "Low",
        "Medium",
        "High",
        "Jam"
    ]
)

vehicle_condition = st.selectbox(
    "Vehicle Condition",
    [0, 1, 2, 3],
    index=2
)

vehicle_type = st.selectbox(
    "Vehicle Type",
    [
        "motorcycle",
        "scooter",
        "electric_scooter",
        "bicycle"
    ]
)

order_type = st.selectbox(
    "Order Type",
    [
        "Snack",
        "Drinks",
        "Buffet",
        "Meal"
    ]
)

city = st.selectbox(
    "City",
    [
        "Urban",
        "Metropolitian",
        "Semi-Urban"
    ]
)

festival = st.selectbox(
    "Festival",
    ["No", "Yes"]
)

multiple_deliveries = st.selectbox(
    "Multiple Deliveries",
    [0.0, 1.0, 2.0, 3.0],
    index=1
)

order_period = st.selectbox(
    "Order Period",
    [
        "Morning",
        "Afternoon",
        "Evening",
        "Night"
    ]
)

order_day = st.selectbox(
    "Order Day",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]
)

order_month = st.selectbox(
    "Order Month",
    [2, 3, 4],
    index=1
)

order_hour = st.selectbox(
    "Order Hour",
    list(range(24)),
    index=18
)

restaurant_latitude = st.number_input(
    "Restaurant Latitude",
    value=20.0,
    format="%.6f"
)

restaurant_longitude = st.number_input(
    "Restaurant Longitude",
    value=85.0,
    format="%.6f"
)

delivery_latitude = st.number_input(
    "Delivery Location Latitude",
    value=20.1,
    format="%.6f"
)

delivery_longitude = st.number_input(
    "Delivery Location Longitude",
    value=85.1,
    format="%.6f"
)



def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.asin(math.sqrt(a))

    return R * c

distance_km = calculate_distance(
    restaurant_latitude,
    restaurant_longitude,
    delivery_latitude,
    delivery_longitude
)

st.info(f"Estimated delivery distance: {distance_km:.2f} km")

order_picked_hour = st.selectbox(
    "Order Picked Hour",
    list(range(24)),
    index=18
)

preparation_time = st.number_input(
    "Preparation Time (minutes)",
    min_value=0.0,
    max_value=20.0,
    value=9.5,
    step=0.5
)

input_data = pd.DataFrame([{
    "Delivery_person_Age": delivery_age,
    "Delivery_person_Ratings": delivery_rating,
    "Restaurant_latitude": restaurant_latitude,
    "Restaurant_longitude": restaurant_longitude,
    "Delivery_location_latitude": delivery_latitude,
    "Delivery_location_longitude": delivery_longitude,
    "Weatherconditions": weather,
    "Road_traffic_density": traffic,
    "Vehicle_condition": vehicle_condition,
    "Type_of_order": order_type,
    "Type_of_vehicle": vehicle_type,
    "multiple_deliveries": multiple_deliveries,
    "Festival": festival,
    "City": city,
    "distance_km": distance_km,
    "Order_Month": order_month,
    "Order_Hour": order_hour,
    "Order_Picked_Hour": order_picked_hour,
    "Order_Period": order_period,
    "Order_Day": order_day,
    "Preparation_Time_min": preparation_time,
    "Suspicious_Age_Rating": delivery_age == 50 and delivery_rating == 6
}])

st.subheader("Model Input")
st.dataframe(input_data)

if st.button("Predict Delivery Time"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Delivery Time: {prediction:.1f} minutes")