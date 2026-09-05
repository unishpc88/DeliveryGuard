import streamlit as st
import joblib
import math
import pandas as pd
import shap

st.title("DeliveryGuard")
st.write("Food Delivery Time Prediction System")

model_path = "models/deliveryguard_model.pkl"
model = joblib.load(model_path)
explainer = shap.TreeExplainer(
    model.named_steps["model"]
)

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

    processed_input = model.named_steps["preprocessor"].transform(input_data)

    shap_values = explainer.shap_values(processed_input)

    local_explanation = pd.DataFrame({
        "Feature": model.named_steps["preprocessor"].get_feature_names_out(),
        "SHAP_Value": shap_values[0]
    })

    local_explanation["Absolute_SHAP"] = (
        local_explanation["SHAP_Value"].abs()
    )

    top_factors = local_explanation.sort_values(
        "Absolute_SHAP",
        ascending=False
    ).head(6)

    feature_labels = {
        "num__distance_km": "📍 Distance",
        "num__Delivery_person_Ratings": "⭐ Delivery Person Rating",
        "num__Delivery_person_Age": "👤 Delivery Person Age",
        "num__Vehicle_condition": "🛵 Vehicle Condition",
        "num__multiple_deliveries": "📦 Multiple Deliveries",
        "cat__Road_traffic_density_Low": "🚦 Low Traffic",
        "cat__Road_traffic_density_Medium": "🚦 Medium Traffic",
        "cat__Road_traffic_density_High": "🚦 High Traffic",
        "cat__Road_traffic_density_Jam": "🚦 Jam Traffic",
        "cat__Weatherconditions_conditions Sunny": "☀️ Sunny Weather",
        "cat__Weatherconditions_conditions Cloudy": "☁️ Cloudy Weather",
        "cat__Weatherconditions_conditions Fog": "🌫️ Foggy Weather",
        "cat__Weatherconditions_conditions Stormy": "⛈️ Stormy Weather",
        "cat__Weatherconditions_conditions Sandstorms": "🌪️ Sandstorms",
        "cat__Weatherconditions_conditions Windy": "💨 Windy Weather",
        "num__Order_Hour": "🕐 Order Hour",
        "num__Order_Picked_Hour": "🕐 Pickup Hour",
        "num__Preparation_Time_min": "🍳 Preparation Time"
    }

    feature_values = {
        "num__distance_km": f"📍 Distance ({distance_km:.1f} km)",
        "num__Delivery_person_Ratings": f"⭐ Delivery Person Rating ({delivery_rating:.1f})",
        "num__Delivery_person_Age": f"👤 Delivery Person Age ({delivery_age:.0f})",
        "num__Vehicle_condition": f"🛵 Vehicle Condition ({vehicle_condition})",
        "num__multiple_deliveries": f"📦 Multiple Deliveries ({multiple_deliveries:.0f})",
        "cat__Road_traffic_density_Low": "🚦 Low Traffic",
        "cat__Road_traffic_density_Medium": "🚦 Medium Traffic",
        "cat__Road_traffic_density_High": "🚦 High Traffic",
        "cat__Road_traffic_density_Jam": "🚦 Jam Traffic",
        "cat__Weatherconditions_conditions Sunny": "☀️ Sunny Weather",
        "cat__Weatherconditions_conditions Cloudy": "☁️ Cloudy Weather",
        "cat__Weatherconditions_conditions Fog": "🌫️ Foggy Weather",
        "cat__Weatherconditions_conditions Stormy": "⛈️ Stormy Weather",
        "cat__Weatherconditions_conditions Sandstorms": "🌪️ Sandstorms",
        "cat__Weatherconditions_conditions Windy": "💨 Windy Weather",
        "num__Order_Hour": f"🕐 Order Hour ({order_hour}:00)",
        "num__Order_Picked_Hour": f"🕐 Pickup Hour ({order_picked_hour}:00)",
        "num__Preparation_Time_min": f"🍳 Preparation Time ({preparation_time:.1f} min)"
    }

    top_factors["Feature"] = top_factors["Feature"].map(
        lambda x: feature_values.get(
            x,
            feature_labels.get(x, x)
        )
    )

    st.subheader("DeliveryGuard Result")

    st.success(
        f"Estimated Delivery Time: {prediction:.1f} minutes"
    )

    if prediction <= 21:
        st.info("🟢 Normal Delivery")
    elif prediction < 30:
        st.warning("🟡 Moderate Delivery Time")
    else:
        st.error("🔴 High Delivery Time")

    st.subheader("🔍 Why this delivery time?")

    increasing = top_factors[
        top_factors["SHAP_Value"] > 0
    ]

    decreasing = top_factors[
        top_factors["SHAP_Value"] < 0
    ]

    if len(increasing) > 0:
        st.markdown("### ⬆️ Factors increasing delivery time")

        for _, row in increasing.iterrows():
            st.write(
                f"{row['Feature']} — increased estimate by "
                f"**{row['SHAP_Value']:.1f} min**"
            )

    if len(decreasing) > 0:
        st.markdown("### ⬇️ Factors reducing delivery time")

        for _, row in decreasing.iterrows():
            st.write(
                f"{row['Feature']} — reduced estimate by "
                f"**{abs(row['SHAP_Value']):.1f} min**"
            )