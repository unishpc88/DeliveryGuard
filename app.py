import streamlit as st
import joblib
import math
import pandas as pd
import shap
import numpy as np

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

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

    # Preprocess input for SHAP and prediction stability
    processed_input = model.named_steps["preprocessor"].transform(input_data)

    # Prediction stability
    tree_predictions = [
        tree.predict(processed_input)[0]
        for tree in model.named_steps["model"].estimators_
    ]

    prediction_std = np.std(tree_predictions)

    # SHAP explanation
    shap_values = explainer.shap_values(processed_input)

    feature_names = model.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Value": shap_values[0]
    })

    shap_df["Abs_SHAP"] = shap_df["SHAP_Value"].abs()

    top_factors = shap_df.sort_values(
        "Abs_SHAP",
        ascending=False
    ).head(6)

    # Human-readable feature names
    feature_labels = {
        "num__Delivery_person_Age": "👤 Delivery Person Age",
        "num__Delivery_person_Ratings": "⭐ Delivery Person Rating",
        "num__distance_km": "📍 Distance",
        "num__Vehicle_condition": "🛵 Vehicle Condition",
        "num__multiple_deliveries": "📦 Multiple Deliveries",
        "num__Preparation_Time_min": "⏱️ Preparation Time",
        "num__Order_Hour": "🕐 Order Hour",
        "num__Order_Picked_Hour": "🕐 Order Picked Hour",
        "num__Suspicious_Age_Rating": "⚠️ Data Quality Flag",
        "cat__Road_traffic_density_High": "🚦 High Traffic",
        "cat__Road_traffic_density_Jam": "🚦 Jam Traffic",
        "cat__Road_traffic_density_Low": "🚦 Low Traffic",
        "cat__Road_traffic_density_Medium": "🚦 Medium Traffic",
        "cat__Weatherconditions_conditions Cloudy": "🌦️ Cloudy Weather",
        "cat__Weatherconditions_conditions Fog": "🌫️ Fog",
        "cat__Weatherconditions_conditions Sandstorms": "🌪️ Sandstorms",
        "cat__Weatherconditions_conditions Stormy": "⛈️ Stormy Weather",
        "cat__Weatherconditions_conditions Sunny": "☀️ Sunny Weather",
        "cat__Weatherconditions_conditions Windy": "💨 Windy Weather",
        "cat__Festival_Yes": "🎉 Festival",
        "cat__Festival_No": "🎉 No Festival",
        "cat__City_Metropolitian": "🏙️ Metropolitan City",
        "cat__City_Urban": "🏙️ Urban City",
        "cat__City_Semi-Urban": "🏙️ Semi-Urban City",
        "cat__Order_Period_Morning": "🌅 Morning",
        "cat__Order_Period_Afternoon": "☀️ Afternoon",
        "cat__Order_Period_Evening": "🌆 Evening",
        "cat__Order_Period_Night": "🌙 Night",
        "cat__Order_Day_Monday": "📅 Monday",
        "cat__Order_Day_Tuesday": "📅 Tuesday",
        "cat__Order_Day_Wednesday": "📅 Wednesday",
        "cat__Order_Day_Thursday": "📅 Thursday",
        "cat__Order_Day_Friday": "📅 Friday",
        "cat__Order_Day_Saturday": "📅 Saturday",
        "cat__Order_Day_Sunday": "📅 Sunday",
        "cat__Type_of_order_Buffet": "🍽️ Buffet Order",
        "cat__Type_of_order_Drinks": "🥤 Drinks Order",
        "cat__Type_of_order_Meal": "🍛 Meal Order",
        "cat__Type_of_order_Snack": "🍔 Snack Order",
        "cat__Type_of_vehicle_bicycle": "🚲 Bicycle",
        "cat__Type_of_vehicle_electric_scooter": "🛴 Electric Scooter",
        "cat__Type_of_vehicle_motorcycle": "🏍️ Motorcycle",
        "cat__Type_of_vehicle_scooter": "🛵 Scooter"
    }

    # Display top SHAP factors
    st.write("### 🔍 What is affecting this prediction?")

    for _, row in top_factors.iterrows():

        feature = row["Feature"]
        shap_value = row["SHAP_Value"]

        display_name = feature_labels.get(
            feature,
            feature.replace("num__", "").replace("cat__", "")
        )

        # Get actual input value
        if feature.startswith("num__"):
            original_feature = feature.replace("num__", "")
            input_value = input_data[original_feature].iloc[0]

            if pd.isna(input_value):
                input_value = "Missing"

        else:
            original_feature = feature.replace("cat__", "")
            input_value = "Yes"

            if "_" in original_feature:
                category_value = original_feature.split("_")[-1]

                if category_value not in [
                    str(x) for x in input_data.columns
                ]:
                    input_value = category_value

        if shap_value > 0:
            st.write(
                f"🔴 {display_name} ({input_value}) "
                f"— increased estimate by {shap_value:.1f} min"
            )
        else:
            st.write(
                f"🟢 {display_name} ({input_value}) "
                f"— reduced estimate by {abs(shap_value):.1f} min"
            )

    # Delivery status
    if prediction <= 21:
        status = "🟢 Normal Delivery"
    elif prediction <= 29:
        status = "🟡 Moderate Delivery Time"
    else:
        status = "🔴 High Delivery Time"

    # Distance category
    if distance_km < 5:
        distance_category = "Short"
    elif distance_km < 10:
        distance_category = "Medium"
    elif distance_km < 15:
        distance_category = "Long"
    else:
        distance_category = "Very Long"

    # Prediction history
    st.session_state.prediction_history.append({
        "Predicted Time": round(prediction, 1),
        "Status": status,
        "Distance (km)": round(distance_km, 1),
        "Distance Category": distance_category,
        "Traffic": traffic,
        "Weather": weather.replace("conditions ", "")
    })

    # Prediction result
    st.success(
        f"Predicted Delivery Time: **{prediction:.1f} minutes**"
    )

    st.info(
        f"Delivery Status: **{status}**"
    )

    # Delivery risk
    if prediction >= 30:
        st.warning(
            "⚠️ High delivery risk detected. "
            "Consider checking traffic, weather, distance, "
            "and multiple-delivery conditions."
        )

    elif prediction >= 22:
        st.info(
            "🟡 Moderate delivery time. "
            "Some conditions may be contributing to the delay."
        )

    else:
        st.success(
            "✅ Delivery conditions look favorable."
        )

    # Smart Risk Factors
    positive_factors = top_factors[
        top_factors["SHAP_Value"] > 0
    ]

    if prediction >= 30 and len(positive_factors) > 0:

        st.write("### 🧠 Why is this delivery high risk?")

        for _, row in positive_factors.head(3).iterrows():

            feature = row["Feature"]
            shap_value = row["SHAP_Value"]

            display_name = feature_labels.get(
                feature,
                feature.replace("num__", "").replace("cat__", "")
            )

            st.write(
                f"🔴 {display_name} "
                f"→ adds {shap_value:.1f} minutes "
                f"to the prediction."
            )

    # SHAP-driven Delivery Recommendations
    st.write("### 💡 Delivery Recommendations")

    recommendation_given = False

    for _, row in positive_factors.sort_values(
        "SHAP_Value",
        ascending=False
    ).iterrows():

        feature = row["Feature"]
        shap_value = row["SHAP_Value"]

        if shap_value <= 0:
            continue

        if feature == "num__multiple_deliveries" and multiple_deliveries >= 2:

            st.write(
                f"📦 Multiple deliveries are adding "
                f"**{shap_value:.1f} minutes**. "
                "Consider reducing multiple deliveries assigned "
                "to the same delivery person."
            )

            recommendation_given = True

        elif feature == "num__distance_km" and distance_km >= 10:

            st.write(
                f"📍 Distance is adding **{shap_value:.1f} minutes**. "
                "Consider assigning a nearby delivery person "
                "when possible."
            )

            recommendation_given = True

        elif feature in [
            "cat__Road_traffic_density_High",
            "cat__Road_traffic_density_Jam"
        ]:

            st.write(
                f"🚦 Traffic conditions are adding "
                f"**{shap_value:.1f} minutes**. "
                "Consider less congested routes or delivery periods."
            )

            recommendation_given = True

        elif feature in [
            "cat__Weatherconditions_conditions Cloudy",
            "cat__Weatherconditions_conditions Fog",
            "cat__Weatherconditions_conditions Stormy",
            "cat__Weatherconditions_conditions Sandstorms",
            "cat__Weatherconditions_conditions Windy"
        ]:

            st.write(
                f"🌦️ Weather conditions are adding "
                f"**{shap_value:.1f} minutes**. "
                "Allow additional delivery time or prioritize "
                "suitable routes."
            )

            recommendation_given = True

        elif feature == "num__Vehicle_condition" and vehicle_condition == 0:

            st.write(
                f"🛵 Vehicle condition is adding "
                f"**{shap_value:.1f} minutes**. "
                "Consider assigning a better-conditioned vehicle."
            )

            recommendation_given = True

        elif feature == "num__Delivery_person_Ratings":

            st.write(
                f"⭐ Delivery person rating is contributing "
                f"**{shap_value:.1f} minutes**. "
                "Consider assigning experienced, highly rated "
                "delivery personnel when possible."
            )

            recommendation_given = True

    if not recommendation_given:
        st.write(
            "✅ No major corrective action is recommended "
            "for the current prediction."
        )
# =========================
# Prediction History
# =========================

if len(st.session_state.prediction_history) > 0:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.subheader("📊 Prediction History")

    st.dataframe(
        history_df,
        hide_index=True
    )


# =========================
# DeliveryGuard Analytics
# =========================

if len(st.session_state.prediction_history) > 0:

    st.subheader("📈 DeliveryGuard Analytics")

    total_predictions = len(history_df)

    average_prediction = history_df["Predicted Time"].mean()

    normal_count = (
        history_df["Status"] == "🟢 Normal Delivery"
    ).sum()

    moderate_count = (
        history_df["Status"] == "🟡 Moderate Delivery Time"
    ).sum()

    high_count = (
        history_df["Status"] == "🔴 High Delivery Time"
    ).sum()

    high_rate = (
        high_count / total_predictions
    ) * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Predictions",
            total_predictions
        )

    with col2:
        st.metric(
            "Average Time",
            f"{average_prediction:.1f} min"
        )

    with col3:
        st.metric(
            "🟢 Normal",
            normal_count
        )

    with col4:
        st.metric(
            "High Delivery Rate",
            f"{high_rate:.1f}%"
        )

    st.write("### 📊 Delivery Status Distribution")

    status_chart = pd.DataFrame({
        "Status": [
            "Normal",
            "Moderate",
            "High"
        ],
        "Count": [
            normal_count,
            moderate_count,
            high_count
        ]
    })

    st.bar_chart(
        status_chart,
        x="Status",
        y="Count"
    )

    st.dataframe(
        status_chart,
        hide_index=True
    )

    st.write("### 🚦 Average Predicted Time by Traffic")

    traffic_analysis = (
        history_df
        .groupby("Traffic")["Predicted Time"]
        .mean()
        .reset_index()
    )

    traffic_analysis["Predicted Time"] = (
        traffic_analysis["Predicted Time"].round(1)
    )

    st.bar_chart(
        traffic_analysis,
        x="Traffic",
        y="Predicted Time"
    )

    st.dataframe(
        traffic_analysis,
        hide_index=True
    )

    st.write("### 📍 Average Predicted Time by Distance")

    distance_analysis = (
        history_df
        .groupby("Distance Category")["Predicted Time"]
        .mean()
        .reindex(
            ["Short", "Medium", "Long", "Very Long"]
        )
        .dropna()
        .reset_index()
    )

    distance_analysis["Predicted Time"] = (
        distance_analysis["Predicted Time"].round(1)
    )

    st.bar_chart(
        distance_analysis,
        x="Distance Category",
        y="Predicted Time"
    )

    st.dataframe(
        distance_analysis,
        hide_index=True
    )
    st.write("### 🌦️ Average Predicted Time by Weather")

    weather_analysis = (
        history_df
        .groupby("Weather")["Predicted Time"]
        .mean()
        .reset_index()
    )

    weather_analysis["Predicted Time"] = (
        weather_analysis["Predicted Time"].round(1)
    )

    st.bar_chart(
        weather_analysis,
        x="Weather",
        y="Predicted Time"
    )

    st.dataframe(
        weather_analysis,
        hide_index=True
    )