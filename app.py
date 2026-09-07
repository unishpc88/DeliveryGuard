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

    # =========================================================
    # Basic What-If Scenario: Reduce Multiple Deliveries to 1
    # =========================================================

    scenario_input = input_data.copy()

    scenario_input["multiple_deliveries"] = 1.0

    scenario_prediction = model.predict(scenario_input)[0]

    st.write("### 🔄 What-If Scenario")

    st.write(
        f"📦 If multiple deliveries are reduced to **1**, "
        f"the predicted delivery time becomes "
        f"**{scenario_prediction:.1f} minutes**."
    )

    time_saved = prediction - scenario_prediction

    if time_saved > 0:
        st.success(
            f"⏱️ Potential time reduction: "
            f"**{time_saved:.1f} minutes**"
        )
    elif time_saved < 0:
        st.info(
            f"⏱️ The alternative scenario is "
            f"{abs(time_saved):.1f} minutes slower."
        )
    else:
        st.info("⏱️ No predicted change in delivery time.")

    # =========================================================
    # Multiple-Delivery Scenario Comparison
    # =========================================================

    scenario_results = []

    for deliveries in [0.0, 1.0, 2.0, 3.0]:

        scenario = input_data.copy()
        scenario["multiple_deliveries"] = deliveries

        scenario_prediction = model.predict(scenario)[0]

        scenario_results.append({
            "Multiple Deliveries": int(deliveries),
            "Predicted Time (min)": round(scenario_prediction, 1)
        })

    scenario_df = pd.DataFrame(scenario_results)

    st.write("### 📊 Multiple-Delivery Scenario Comparison")

    st.dataframe(
        scenario_df,
        hide_index=True
    )

    st.line_chart(
        scenario_df,
        x="Multiple Deliveries",
        y="Predicted Time (min)"
    )

    # =========================================================
    # Traffic Scenario Comparison
    # =========================================================

    traffic_results = []

    for traffic_scenario in ["Low", "Medium", "High", "Jam"]:

        traffic_input = input_data.copy()
        traffic_input["Road_traffic_density"] = traffic_scenario

        traffic_prediction = model.predict(traffic_input)[0]

        traffic_results.append({
            "Traffic Condition": traffic_scenario,
            "Predicted Time (min)": round(traffic_prediction, 1)
        })

    traffic_df = pd.DataFrame(traffic_results)

    st.write("### 🚦 Traffic Scenario Comparison")

    st.dataframe(
        traffic_df,
        hide_index=True
    )

    st.line_chart(
        traffic_df,
        x="Traffic Condition",
        y="Predicted Time (min)"
    )

    # =========================================================
    # Weather Scenario Comparison
    # =========================================================

    weather_results = []

    for weather_scenario in [
        "conditions Sunny",
        "conditions Cloudy",
        "conditions Fog",
        "conditions Stormy",
        "conditions Sandstorms",
        "conditions Windy"
    ]:

        weather_input = input_data.copy()
        weather_input["Weatherconditions"] = weather_scenario

        weather_prediction = model.predict(weather_input)[0]

        weather_results.append({
            "Weather Condition": weather_scenario.replace(
                "conditions ", ""
            ),
            "Predicted Time (min)": round(weather_prediction, 1)
        })

    weather_df = pd.DataFrame(weather_results)

    st.write("### 🌦️ Weather Scenario Comparison")

    st.dataframe(
        weather_df,
        hide_index=True
    )

    st.bar_chart(
        weather_df,
        x="Weather Condition",
        y="Predicted Time (min)"
    )

    # =========================================================
    # Distance Scenario Comparison
    # =========================================================

    distance_results = []

    restaurant_lat = input_data[
        "Restaurant_latitude"
    ].iloc[0]

    restaurant_lon = input_data[
        "Restaurant_longitude"
    ].iloc[0]

    current_delivery_lat = input_data[
        "Delivery_location_latitude"
    ].iloc[0]

    current_delivery_lon = input_data[
        "Delivery_location_longitude"
    ].iloc[0]

    # Calculate current direction from restaurant
    # to delivery location

    lat_difference = (
        current_delivery_lat - restaurant_lat
    )

    lon_difference = (
        current_delivery_lon - restaurant_lon
    )

    direction_norm = np.sqrt(
        lat_difference ** 2 +
        lon_difference ** 2
    )

    # Avoid division by zero

    if direction_norm == 0:
        direction_lat = 1
        direction_lon = 0
    else:
        direction_lat = (
            lat_difference / direction_norm
        )

        direction_lon = (
            lon_difference / direction_norm
        )

    for target_distance in [5, 10, 15, 20]:

        distance_input = input_data.copy()

        # Approximate conversion from km
        # to latitude/longitude degrees

        lat_offset = (
            target_distance *
            direction_lat / 111
        )

        lon_offset = (
            target_distance *
            direction_lon /
            (
                111 *
                np.cos(
                    np.radians(restaurant_lat)
                )
            )
        )

        new_delivery_lat = (
            restaurant_lat + lat_offset
        )

        new_delivery_lon = (
            restaurant_lon + lon_offset
        )

        distance_input[
            "Delivery_location_latitude"
        ] = new_delivery_lat

        distance_input[
            "Delivery_location_longitude"
        ] = new_delivery_lon

        distance_input[
            "distance_km"
        ] = target_distance

        distance_prediction = (
            model.predict(distance_input)[0]
        )

        distance_results.append({
            "Distance (km)": target_distance,
            "Predicted Time (min)": round(
                distance_prediction,
                1
            )
        })

    distance_df = pd.DataFrame(
        distance_results
    )

    st.write("### 📍 Distance Scenario Comparison")

    st.dataframe(
        distance_df,
        hide_index=True
    )

    st.line_chart(
        distance_df,
        x="Distance (km)",
        y="Predicted Time (min)"
    )

    # =========================================================
    # Combined Current vs Improved Scenario
    # =========================================================

    improved_input = input_data.copy()

    # Reduce multiple deliveries
    improved_input[
        "multiple_deliveries"
    ] = 1.0

    # Use less congested traffic
    improved_input[
        "Road_traffic_density"
    ] = "Low"

    # Use favorable weather
    improved_input[
        "Weatherconditions"
    ] = "conditions Sunny"

    # ---------------------------------------------------------
    # Set improved distance to 10 km while maintaining
    # approximately the same direction
    # ---------------------------------------------------------

    improved_distance = 10

    improved_lat_offset = (
        improved_distance *
        direction_lat / 111
    )

    improved_lon_offset = (
        improved_distance *
        direction_lon /
        (
            111 *
            np.cos(
                np.radians(restaurant_lat)
            )
        )
    )

    improved_input[
        "Delivery_location_latitude"
    ] = (
        restaurant_lat +
        improved_lat_offset
    )

    improved_input[
        "Delivery_location_longitude"
    ] = (
        restaurant_lon +
        improved_lon_offset
    )

    improved_input[
        "distance_km"
    ] = improved_distance

    improved_prediction = (
        model.predict(improved_input)[0]
    )

    combined_time_saved = (
        prediction -
        improved_prediction
    )

    st.write(
        "### 🚀 Current vs Improved Scenario"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Current Prediction",
            f"{prediction:.1f} min"
        )

    with col2:
        st.metric(
            "Improved Scenario",
            f"{improved_prediction:.1f} min"
        )

    if combined_time_saved > 0:
        st.success(
            f"⏱️ Potential time reduction: "
            f"**{combined_time_saved:.1f} minutes**"
        )
    elif combined_time_saved < 0:
        st.info(
            f"⏱️ The improved scenario is "
            f"{abs(combined_time_saved):.1f} minutes slower."
        )
    else:
        st.info(
            "⏱️ No predicted change between "
            "the two scenarios."
        )

    st.write(
        "📋 Improved scenario: "
        "**1 multiple delivery, Low traffic, "
        "Sunny weather, 10 km distance**."
    )

    # =========================================================
    # Preprocess input for SHAP and prediction stability
    # =========================================================

    processed_input = (
        model.named_steps[
            "preprocessor"
        ].transform(input_data)
    )

    # =========================================================
    # Prediction Stability
    # =========================================================

    tree_predictions = [
        tree.predict(processed_input)[0]
        for tree in model.named_steps[
            "model"
        ].estimators_
    ]

    prediction_std = np.std(
        tree_predictions
    )

    # =========================================================
    # SHAP Explanation
    # =========================================================

    shap_values = explainer.shap_values(
        processed_input
    )

    feature_names = (
        model.named_steps[
            "preprocessor"
        ].get_feature_names_out()
    )

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Value": shap_values[0]
    })

    shap_df["Abs_SHAP"] = (
        shap_df["SHAP_Value"].abs()
    )

    top_factors = (
        shap_df
        .sort_values(
            "Abs_SHAP",
            ascending=False
        )
        .head(6)
    )

    # =========================================================
    # Human-readable feature names
    # =========================================================

    feature_labels = {
        "num__Delivery_person_Age":
            "👤 Delivery Person Age",

        "num__Delivery_person_Ratings":
            "⭐ Delivery Person Rating",

        "num__distance_km":
            "📍 Distance",

        "num__Vehicle_condition":
            "🛵 Vehicle Condition",

        "num__multiple_deliveries":
            "📦 Multiple Deliveries",

        "num__Preparation_Time_min":
            "⏱️ Preparation Time",

        "num__Order_Hour":
            "🕐 Order Hour",

        "num__Order_Picked_Hour":
            "🕐 Order Picked Hour",

        "num__Suspicious_Age_Rating":
            "⚠️ Data Quality Flag",

        "cat__Road_traffic_density_High":
            "🚦 High Traffic",

        "cat__Road_traffic_density_Jam":
            "🚦 Jam Traffic",

        "cat__Road_traffic_density_Low":
            "🚦 Low Traffic",

        "cat__Road_traffic_density_Medium":
            "🚦 Medium Traffic",

        "cat__Weatherconditions_conditions Cloudy":
            "🌦️ Cloudy Weather",

        "cat__Weatherconditions_conditions Fog":
            "🌫️ Fog",

        "cat__Weatherconditions_conditions Sandstorms":
            "🌪️ Sandstorms",

        "cat__Weatherconditions_conditions Stormy":
            "⛈️ Stormy Weather",

        "cat__Weatherconditions_conditions Sunny":
            "☀️ Sunny Weather",

        "cat__Weatherconditions_conditions Windy":
            "💨 Windy Weather",

        "cat__Festival_Yes":
            "🎉 Festival",

        "cat__Festival_No":
            "🎉 No Festival",

        "cat__City_Metropolitian":
            "🏙️ Metropolitan City",

        "cat__City_Urban":
            "🏙️ Urban City",

        "cat__City_Semi-Urban":
            "🏙️ Semi-Urban City",

        "cat__Order_Period_Morning":
            "🌅 Morning",

        "cat__Order_Period_Afternoon":
            "☀️ Afternoon",

        "cat__Order_Period_Evening":
            "🌆 Evening",

        "cat__Order_Period_Night":
            "🌙 Night",

        "cat__Order_Day_Monday":
            "📅 Monday",

        "cat__Order_Day_Tuesday":
            "📅 Tuesday",

        "cat__Order_Day_Wednesday":
            "📅 Wednesday",

        "cat__Order_Day_Thursday":
            "📅 Thursday",

        "cat__Order_Day_Friday":
            "📅 Friday",

        "cat__Order_Day_Saturday":
            "📅 Saturday",

        "cat__Order_Day_Sunday":
            "📅 Sunday",

        "cat__Type_of_order_Buffet":
            "🍽️ Buffet Order",

        "cat__Type_of_order_Drinks":
            "🥤 Drinks Order",

        "cat__Type_of_order_Meal":
            "🍛 Meal Order",

        "cat__Type_of_order_Snack":
            "🍔 Snack Order",

        "cat__Type_of_vehicle_bicycle":
            "🚲 Bicycle",

        "cat__Type_of_vehicle_electric_scooter":
            "🛴 Electric Scooter",

        "cat__Type_of_vehicle_motorcycle":
            "🏍️ Motorcycle",

        "cat__Type_of_vehicle_scooter":
            "🛵 Scooter"
    }

    # =========================================================
    # Display Top SHAP Factors
    # =========================================================

    st.write(
        "### 🔍 What is affecting this prediction?"
    )

    for _, row in top_factors.iterrows():

        feature = row["Feature"]
        shap_value = row["SHAP_Value"]

        display_name = feature_labels.get(
            feature,
            feature
            .replace("num__", "")
            .replace("cat__", "")
        )

        # Get actual input value

        if feature.startswith("num__"):

            original_feature = (
                feature.replace(
                    "num__",
                    ""
                )
            )

            input_value = (
                input_data[
                    original_feature
                ].iloc[0]
            )

            if pd.isna(input_value):
                input_value = "Missing"

        else:

            original_feature = (
                feature.replace(
                    "cat__",
                    ""
                )
            )

            input_value = "Yes"

            if "_" in original_feature:

                category_value = (
                    original_feature
                    .split("_")[-1]
                )

                if category_value not in [
                    str(x)
                    for x in input_data.columns
                ]:
                    input_value = category_value

        if shap_value > 0:

            st.write(
                f"🔴 {display_name} "
                f"({input_value}) "
                f"— increased estimate by "
                f"{shap_value:.1f} min"
            )

        else:

            st.write(
                f"🟢 {display_name} "
                f"({input_value}) "
                f"— reduced estimate by "
                f"{abs(shap_value):.1f} min"
            )

    # =========================================================
    # Delivery Status
    # =========================================================

    if prediction <= 21:

        status = "🟢 Normal Delivery"

    elif prediction <= 29:

        status = "🟡 Moderate Delivery Time"

    else:

        status = "🔴 High Delivery Time"

    # =========================================================
    # Distance Category
    # =========================================================

    if distance_km < 5:

        distance_category = "Short"

    elif distance_km < 10:

        distance_category = "Medium"

    elif distance_km < 15:

        distance_category = "Long"

    else:

        distance_category = "Very Long"

    # =========================================================
    # Prediction History
    # =========================================================

    st.session_state.prediction_history.append({
        "Predicted Time":
            round(prediction, 1),

        "Status":
            status,

        "Distance (km)":
            round(distance_km, 1),

        "Distance Category":
            distance_category,

        "Traffic":
            traffic,

        "Weather":
            weather.replace(
                "conditions ",
                ""
            )
    })

    # =========================================================
    # Prediction Result
    # =========================================================

    st.success(
        f"Predicted Delivery Time: "
        f"**{prediction:.1f} minutes**"
    )

    st.info(
        f"Delivery Status: **{status}**"
    )

    # =========================================================
    # Delivery Risk
    # =========================================================

    if prediction >= 30:

        st.warning(
            "⚠️ High delivery risk detected. "
            "Consider checking traffic, weather, "
            "distance, and multiple-delivery conditions."
        )

    elif prediction >= 22:

        st.info(
            "🟡 Moderate delivery time. "
            "Some conditions may be contributing "
            "to the delay."
        )

    else:

        st.success(
            "✅ Delivery conditions look favorable."
        )

    # =========================================================
    # Smart Risk Factors
    # =========================================================

    positive_factors = top_factors[
        top_factors["SHAP_Value"] > 0
    ]

    if (
        prediction >= 30
        and len(positive_factors) > 0
    ):

        st.write(
            "### 🧠 Why is this delivery high risk?"
        )

        for _, row in (
            positive_factors
            .head(3)
            .iterrows()
        ):

            feature = row["Feature"]
            shap_value = row["SHAP_Value"]

            display_name = feature_labels.get(
                feature,
                feature
                .replace("num__", "")
                .replace("cat__", "")
            )

            st.write(
                f"🔴 {display_name} "
                f"→ adds {shap_value:.1f} minutes "
                f"to the prediction."
            )

    # =========================================================
    # SHAP-driven Delivery Recommendations
    # =========================================================

    st.write(
        "### 💡 Delivery Recommendations"
    )

    recommendation_given = False

    for _, row in (
        positive_factors
        .sort_values(
            "SHAP_Value",
            ascending=False
        )
        .iterrows()
    ):

        feature = row["Feature"]
        shap_value = row["SHAP_Value"]

        if shap_value <= 0:
            continue

        if (
            feature ==
            "num__multiple_deliveries"
            and multiple_deliveries >= 2
        ):

            st.write(
                f"📦 Multiple deliveries are adding "
                f"**{shap_value:.1f} minutes**. "
                "Consider reducing multiple deliveries "
                "assigned to the same delivery person."
            )

            recommendation_given = True

        elif (
            feature ==
            "num__distance_km"
            and distance_km >= 10
        ):

            st.write(
                f"📍 Distance is adding "
                f"**{shap_value:.1f} minutes**. "
                "Consider assigning a nearby delivery "
                "person when possible."
            )

            recommendation_given = True

        elif feature in [
            "cat__Road_traffic_density_High",
            "cat__Road_traffic_density_Jam"
        ]:

            st.write(
                f"🚦 Traffic conditions are adding "
                f"**{shap_value:.1f} minutes**. "
                "Consider less congested routes "
                "or delivery periods."
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
                "Allow additional delivery time "
                "or prioritize suitable routes."
            )

            recommendation_given = True

        elif (
            feature ==
            "num__Vehicle_condition"
            and vehicle_condition == 0
        ):

            st.write(
                f"🛵 Vehicle condition is adding "
                f"**{shap_value:.1f} minutes**. "
                "Consider assigning a "
                "better-conditioned vehicle."
            )

            recommendation_given = True

        elif (
            feature ==
            "num__Delivery_person_Ratings"
        ):

            st.write(
                f"⭐ Delivery person rating is "
                f"contributing **{shap_value:.1f} minutes**. "
                "Consider assigning experienced, "
                "highly rated delivery personnel "
                "when possible."
            )

            recommendation_given = True

    if not recommendation_given:

        st.write(
            "✅ No major corrective action is "
            "recommended for the current prediction."
        )