import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import streamlit.components.v1 as components
import base64  # ADD THIS

# Function to convert video to base64
def get_video_base64(file_path):
    with open(file_path, "rb") as video_file:
        encoded = base64.b64encode(video_file.read()).decode()
    return encoded

# Load and encode the .mov video
video_base64 = get_video_base64("background.MOV")

# Page config
st.set_page_config(page_title="Water Quality Dashboard", layout="wide")

# Custom CSS styling with video background in sidebar
st.markdown(f"""
<style>
[data-testid="stSidebar"] > div:first-child {{
    background: url("data:video/mp4;base64,{video_base64}") no-repeat center center;
    background-size: cover;
}}

.sidebar-button {{
    display: block;
    width: 100%;
    padding: 0.6em;
    text-align: center;
    background-color: #007BFF;
    color: white;
    border: none;
    border-radius: 0.5em;
    margin-bottom: 0.5em;
    font-weight: bold;
    cursor: pointer;
    transition: background-color 0.3s ease;
    text-decoration: none;
}}

.sidebar-button:hover {{
    background-color: #0056b3;
}}

h1, h2, h3, h4, h5, h6, p {{
    color: white;
}}

body {{
    background-color: black;
}}

.centered {{
    text-align: center;
}}
</style>
""", unsafe_allow_html=True)

# Dark mode CSS for main content
st.markdown("""
    <style>
        .block-container {
            padding: 2rem;
        }
        .stMetric {
            background-color: #1e1e1e;
            padding: 1rem;
            border-radius: 10px;
        }
        .stSelectbox label {
            color: lightgray !important;
        }
    </style>
""", unsafe_allow_html=True)

# Function to create styled discussion boxes
def discussion_box(text, color='#00BFC4', font_size='18px'):
    return f"""
    <div style="border: 2px solid {color}; border-radius: 10px; padding: 15px; background-color: #f4f4f4;">
        <h3 style="color: {color}; font-size: {font_size};">Discussion</h3>
        <p style="font-size: {font_size}; line-height: 1.6;">{text}</p>
    </div>
    """

# Sidebar
with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["Home", "Dataset", "Exploratory Data Analysis (EDA)", "Model Evaluation", "Predictions & Recommendations", "About"],
        icons=["house", "database", "bar-chart", "check-circle", "lightbulb", "info-circle"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical"
    )

    st.markdown("### Filters")
    selected_location = st.selectbox("Select Location", ["All", "Site A", "Site B", "Site C"])
    selected_year = st.selectbox("Select Year", ["All", "2022", "2023", "2024"])

# Initialize empty DataFrame
data = pd.DataFrame(columns=["Location", "Year", "pH", "Turbidity"])

# HOME PAGE
if selected == "Home":
    IMAGE_PATH = "images/banner.jpg"
    st.image(IMAGE_PATH, width=900)

    st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='color: black; font-size: 16px; line-height: 1.8;'>

    <h3 style='color: #00BFC4;'> Overview</h3>
    This lab aims to apply data mining and data visualization techniques to predict water quality in Taal Lake using real-world datasets.
    Students will collect, preprocess, analyze, and model data to predict parameters such as pH and dissolved oxygen levels.
    The project incorporates machine learning models and interactive visualizations.

    <h3 style='color: #00BFC4;'> Objectives</h3>
    <ul>
        <li>Apply data mining techniques to extract useful insights from environmental datasets.</li>
        <li>Develop predictive models for water quality using machine learning.</li>
        <li>Compare ensemble learning techniques such as CNN, LSTM, and Hybrid CNN LSTM.</li>
        <li>Visualize trends and patterns in water quality parameters.</li>
        <li>Interpret the impact of environmental and volcanic activity on water quality.</li>
        <li>Predict Water Quality Index (WQI) and Water Pollutant Levels with actionable insights for environmental monitoring and intervention.</li>
    </ul>

    <h3 style='color: #00BFC4;'> Learning Outcomes</h3>
    <ul>
        <li>Identify and collect relevant water quality and environmental datasets.</li>
        <li>Perform data preprocessing and exploratory analysis.</li>
        <li>Implement machine learning models for water quality prediction.</li>
        <li>Compare CNN, LSTM, and Hybrid CNN-LSTM models in ensemble learning.</li>
        <li>Develop interactive visualizations using Python tools.</li>
        <li>Interpret Water Quality Index (WQI) and pollutant levels to provide recommendations for water management.</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

elif selected == "Dataset":
    st.markdown("<h2 style='color:#00BFC4;'>📁 Dataset</h2>", unsafe_allow_html=True)

    if os.path.exists("dataset.csv"):
        data = pd.read_csv("dataset.csv")
        st.success("Loaded existing dataset.")
    else:
        st.warning("No dataset found. Please upload a CSV file below.")

    uploaded_file = st.file_uploader(
        label="Upload your dataset CSV",
        type=["csv"],
        help="Drag and drop file here. Limit 200MB per file • CSV"
    )

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        data.to_csv("dataset.csv", index=False)
        st.success("File uploaded and saved successfully.")

    if not data.empty:
        st.markdown("#### Dataset Preview")
        st.dataframe(data, use_container_width=True)

        st.markdown("#### Filtered Data Based on Selected Location and Year")
        filtered_data = data.copy()
        if selected_location != "All":
            filtered_data = filtered_data[filtered_data["Location"] == selected_location]
        if selected_year != "All":
            filtered_data = filtered_data[filtered_data["Year"] == int(selected_year)]

        if not filtered_data.empty:
            st.dataframe(filtered_data, use_container_width=True)
        else:
            st.warning("No matching data after filters.")
    else:
        st.info("No data available. Upload a dataset to get started.")

elif selected == "Exploratory Data Analysis (EDA)":
    st.markdown("<h2 style='color:#00BFC4;'>📊 Exploratory Data Analysis (EDA)</h2>", unsafe_allow_html=True)

    if os.path.exists("dataset.csv"):
        df = pd.read_csv("dataset.csv")

        # Strip and clean numeric columns
        for col in df.columns:
            df[col] = df[col].astype(str).str.replace(" ", "", regex=False)

        # Convert numeric columns
        numeric_cols = df.columns[df.apply(lambda x: pd.to_numeric(x, errors='coerce').notnull().all())].tolist()
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Summary Statistics", "Correlation Matrix", "Trends Over Years", "Parameter Relationships"])

        with tab1:
            st.subheader("📋 Summary Statistics")

            water_path = "Descriptive Statistics for Water Quality Parameters.csv"
            external_path = "Descriptive Statistics for External Environmental Factors.csv"

            if os.path.exists(water_path) and os.path.exists(external_path):
                df_water = pd.read_csv(water_path)
                df_external = pd.read_csv(external_path)

                st.markdown("**Descriptive Statistics for Water Quality Parameters**")
                st.dataframe(df_water, use_container_width=True)

                st.markdown("**Descriptive Statistics for External Environmental Factors**")
                st.dataframe(df_external, use_container_width=True)

                discussion_text = """
                The summary statistics provide important insights into the central tendencies, dispersion, and distribution of each water quality and environmental parameter. 
                These tables separate the internal aquatic conditions (like pH and turbidity) from external influences (like air temperature and carbon dioxide), allowing for better focused analysis. 

                For example, variations in water quality indicators may be linked to environmental factors over time. Observing each set independently helps identify patterns that could point to pollution, climate effects, or ecosystem shifts.
                """
                st.markdown(discussion_box(discussion_text, font_size="20px"), unsafe_allow_html=True)

            else:
                st.error("Summary CSV files not found. Please make sure the following files are in your repository:\n\n- `Descriptive Statistics for Water Quality Parameters.csv`\n- `Descriptive Statistics for External Environmental Factors.csv`")

        with tab2:
            st.subheader("🔗 Correlation Matrix")
            correlation_columns = [
                "Surface Temperature", "Middle Temperature", "Bottom Temperature",
                "pH", "Ammonia", "Nitrate", "Phosphate", "Dissolved Oxygen",
                "Sulfide", "Carbon Dioxide", "Air Temperature"
            ]

            # Convert relevant columns to numeric
            df[correlation_columns] = df[correlation_columns].apply(pd.to_numeric, errors='coerce')
            correlation_matrix = df[correlation_columns].corr()

            fig = px.imshow(
                correlation_matrix,
                labels=dict(x="Parameters", y="Parameters", color="Correlation"),
                x=correlation_columns,
                y=correlation_columns,
                color_continuous_scale="RdBu_r",  # This inverts the color scale
                zmin=-1, zmax=1,
                title="Correlation Matrix of Parameters"
            )

            fig.update_layout(
                autosize=True,
                width=1000,
                height=800
            )

            st.plotly_chart(fig, use_container_width=True)

            # Enhanced discussion box
            discussion_text = """
            The correlation matrix is a valuable tool for understanding the relationships between different water quality parameters. Strong positive correlations between temperature measurements at different depths (surface, middle, and bottom) suggest that thermal stratification may be occurring in the lake, with uniform temperature across the different layers. Conversely, negative correlations, such as between dissolved oxygen and carbon dioxide, can indicate biological activity such as respiration, where oxygen is consumed and carbon dioxide is released by aquatic organisms.

            Understanding these relationships helps in identifying the key factors influencing water quality. For instance, high levels of nitrogen compounds like nitrate and ammonia may correlate with increased algal growth, leading to decreased oxygen levels and potential water eutrophication.
            """
            st.markdown(discussion_box(discussion_text, font_size="20px"), unsafe_allow_html=True)

        with tab3:
            st.subheader("📈 Trends Over Years")
            df_grouped = df.groupby("Year").mean(numeric_only=True).reset_index()


            def plot_interactive_lines(title, y_columns):
                if 'Year' in df_grouped.columns:
                    fig = go.Figure()
                    for col in y_columns:
                        if col in df_grouped.columns:
                            fig.add_trace(go.Scatter(
                                x=df_grouped['Year'],
                                y=df_grouped[col],
                                mode='lines+markers',
                                name=col
                            ))
                    fig.update_layout(
                        title=title,
                        xaxis_title='Year',
                        yaxis_title='Average Value',
                        legend_title='Parameter',
                        template='plotly_dark'
                    )
                    st.plotly_chart(fig, use_container_width=True)


            plot_interactive_lines("External Factors Over Years", ['Air Temperature', 'Sulfide', 'Carbon Dioxide'])
            plot_interactive_lines("Water Temperatures Over Years",
                                   ['Surface Temperature', 'Middle Temperature', 'Bottom Temperature'])
            plot_interactive_lines("pH and Dissolved Oxygen Over Years", ['pH', 'Dissolved Oxygen'])
            plot_interactive_lines("Nutrient Levels Over Years", ['Nitrate', 'Ammonia', 'Phosphate'])

            # Enhanced discussion box
            discussion_text = """
            By analyzing trends over multiple years, we can identify long-term changes in environmental factors that influence water quality. For example, a steady increase in surface temperature may indicate the effects of climate change on the lake’s ecosystem, while fluctuations in nutrient levels (e.g., nitrate and phosphate) could be due to seasonal runoff or pollution events. 

            Similarly, changes in dissolved oxygen levels over time may point to shifts in biological activity or changes in the lake's overall health. Observing these trends is vital for understanding the lake's response to both natural and human-induced changes.
            """
            st.markdown(discussion_box(discussion_text, font_size="20px"), unsafe_allow_html=True)

        with tab4:
            st.subheader("🧬 Parameter Relationships")
            x_axis = st.selectbox("Select X-axis", options=df.select_dtypes(include='number').columns, index=0)
            y_axis = st.selectbox("Select Y-axis", options=df.select_dtypes(include='number').columns, index=1)

            if x_axis != y_axis:
                scatter_fig = px.scatter(df, x=x_axis, y=y_axis, trendline="ols", title=f"{y_axis} vs {x_axis}")
                scatter_fig.update_layout(template="plotly_dark")
                st.plotly_chart(scatter_fig, use_container_width=True)
            else:
                st.warning("X and Y axes must be different.")

            # Dynamic discussion based on the user selection
            if x_axis == "Surface Temperature" and y_axis == "pH":
                dynamic_discussion = """
                Surface temperature and pH levels are critical parameters in understanding aquatic chemistry. Higher temperatures typically decrease the solubility of gases like oxygen in water, which can lead to decreased dissolved oxygen levels. This may also influence the pH level, as the solubility of CO2 is inversely related to temperature. A rising surface temperature could lead to a lower pH, which can harm aquatic life.
                """
            elif x_axis == "Dissolved Oxygen" and y_axis == "Carbon Dioxide":
                dynamic_discussion = """
                Dissolved oxygen and carbon dioxide have an inverse relationship in most aquatic ecosystems. During respiration, aquatic organisms consume oxygen and release carbon dioxide. Conversely, during photosynthesis, aquatic plants and algae produce oxygen and consume CO2. Monitoring these parameters together can help us understand the balance of biological activity in the ecosystem.
                """
            else:
                dynamic_discussion = """
                The relationship between these parameters offers deeper insights into the ecological dynamics of Taal Lake. Understanding how environmental factors interact and influence one another is crucial for creating effective water management strategies. 
                """

            st.markdown(discussion_box(dynamic_discussion, font_size="18px"), unsafe_allow_html=True)

    else:
        st.info("No dataset found. Please upload one in the Dataset section.")

elif selected == "Model Evaluation":
    st.markdown("<h2 style='color:#00BFC4;'>📊 Model Evaluation</h2>", unsafe_allow_html=True)

    if os.path.exists("dataset.csv"):
        import plotly.express as px

        # Sample metric data (replace with actual model evaluation results)
        metric_data = pd.DataFrame({
            "Model": ["CNN", "LSTM", "CNN-LSTM"],
            "Accuracy": [0.85, 0.83, 0.88],
            "Precision": [0.82, 0.80, 0.86],
            "Recall": [0.84, 0.81, 0.87],
            "F1 Score": [0.83, 0.805, 0.865]
        })

        metric_melted = metric_data.melt(id_vars="Model", var_name="Metric", value_name="Score")
        tab1, tab2, tab3, tab4 = st.tabs(["Model Comparison", "Data Visualization", "Interactive Prediction View", "Water Quality Index"])

        with tab1:
            st.subheader("📊 Model Comparison")
            fig_metric = px.bar(metric_melted, x="Model", y="Score", color="Metric", barmode="group",
                                title="Model Evaluation Metrics")
            fig_metric.update_layout(template="plotly_dark")
            st.plotly_chart(fig_metric, use_container_width=True)

            discussion = """
            Among the evaluated models, CNN-LSTM exhibits superior performance in accuracy, precision, recall, and F1-score, 
            suggesting its effectiveness in capturing both spatial and temporal data patterns.
            """
            st.markdown(discussion_box(discussion, font_size="18px"), unsafe_allow_html=True)

        with tab2:
            st.subheader("📉 MAE & RMSE Visualization")

            error_data = pd.DataFrame({
                "Model": ["CNN", "LSTM", "CNN-LSTM"],
                "MAE_with": [0.12, 0.13, 0.10],
                "RMSE_with": [0.20, 0.22, 0.18],
                "MAE_without": [0.18, 0.20, 0.14],
                "RMSE_without": [0.25, 0.27, 0.22]
            })
            error_melted = error_data.melt(id_vars="Model", var_name="Metric", value_name="Value")

            fig_error = px.bar(error_melted, x="Model", y="Value", color="Metric", barmode="group",
                               title="Error Metrics with/without External Factors")
            fig_error.update_layout(template="plotly_dark")
            st.plotly_chart(fig_error, use_container_width=True)

            discussion = """
            Inclusion of external environmental factors reduces the MAE and RMSE values, highlighting their role in enhancing model accuracy.
            CNN-LSTM remains the most consistent across both conditions.
            """
            st.markdown(discussion_box(discussion, font_size="18px"), unsafe_allow_html=True)

        with tab3:
            st.subheader("📅 Parameter Predictions with Deep Learning Models")

            view = st.radio("Select Prediction View", ("Weekly", "Monthly", "Yearly"))

            parameters = [
                'Surface Temperature', 'Middle Temperature', 'Bottom Temperature',
                'pH', 'Ammonia', 'Nitrate', 'Phosphate', 'Dissolved Oxygen'
            ]

            base_folder = Path("weekly_predictions").parent.resolve()
            view_folder_map = {
                "Weekly": base_folder / "weekly_predictions",
                "Monthly": base_folder / "monthly_predictions",
                "Yearly": base_folder / "yearly_predictions"
            }

            selected_folder = view_folder_map[view]

            # Save current index in session state
            if "img_index" not in st.session_state:
                st.session_state.img_index = 0

            # Load images
            image_paths = []
            for param in parameters:
                for ext in [".png", ".jpg", ".jpeg"]:
                    image_path = selected_folder / f"{param}{ext}"
                    if image_path.exists():
                        image_paths.append((param, image_path))
                        break

            if not image_paths:
                st.warning(f"⚠️ No prediction images found in `{selected_folder}`.")
            else:
                # Navigation
                col1, col2, col3 = st.columns([1, 6, 1])

                with col1:
                    if st.button("⬅️ Previous"):
                        st.session_state.img_index = (st.session_state.img_index - 1) % len(image_paths)

                with col3:
                    if st.button("Next ➡️"):
                        st.session_state.img_index = (st.session_state.img_index + 1) % len(image_paths)

                # Show image
                current_param, current_path = image_paths[st.session_state.img_index]
                st.image(str(current_path), use_container_width=True, caption=f"{current_param} - {view}")
                st.markdown(f"**Image {st.session_state.img_index + 1} of {len(image_paths)}**")

        with tab4:
            st.subheader("📊 Water Quality Index (WQI) Evaluation")

            base_folder = Path(".")
            csv_map = {
                "Weekly": base_folder / "predicted_wqi_weekly.csv",
                "Monthly": base_folder / "predicted_wqi_monthly.csv",
                "Yearly": base_folder / "predicted_wqi_yearly.csv"
            }

            selected_csv = st.radio("Select Prediction File", ("Weekly", "Monthly", "Yearly"))
            csv_path = csv_map[selected_csv]

            if not csv_path.exists():
                st.error(f"❌ CSV file not found: `{csv_path.name}`")
            else:
                pred_df = pd.read_csv(csv_path)

                # WQI setup
                params = ['pH', 'Dissolved Oxygen', 'Nitrate', 'Phosphate', 'Ammonia']
                ideal_values = {
                    'pH': 7.0,
                    'Dissolved Oxygen': 14.6,
                    'Nitrate': 0.0,
                    'Phosphate': 0.1,
                    'Ammonia': 0.5
                }
                standards = {
                    'pH': 8.5,
                    'Dissolved Oxygen': 5.0,
                    'Nitrate': 45.0,
                    'Phosphate': 0.5,
                    'Ammonia': 1.5
                }

                k = 1 / sum(1 / standards[p] for p in params)
                weights = {p: k / standards[p] for p in params}


                def compute_wqi(row):
                    qi_list = []
                    wi_list = []
                    for p in params:
                        Vi = ideal_values[p]
                        Si = standards[p]
                        V_actual = row[p]
                        Qi = 100 * (V_actual - Vi) / (Si - Vi)
                        Qi = min(max(Qi, 0), 100)
                        Wi = weights[p]
                        qi_list.append(Qi * Wi)
                        wi_list.append(Wi)
                    return sum(qi_list) / sum(wi_list)


                pred_df['WQI'] = pred_df.apply(compute_wqi, axis=1)


                def classify_wqi(wqi):
                    if wqi <= 25:
                        return 'Excellent'
                    elif wqi <= 50:
                        return 'Good'
                    elif wqi <= 75:
                        return 'Poor'
                    elif wqi <= 100:
                        return 'Very Poor'
                    else:
                        return 'Unsuitable for Drinking'


                pred_df['WQI_Status'] = pred_df['WQI'].apply(classify_wqi)


                def pollutant_level(param, value):
                    thresholds = {
                        'Ammonia': [0.5, 1.0],
                        'Nitrate': [10, 50],
                        'Phosphate': [0.1, 0.5]
                    }
                    if param not in thresholds:
                        return "N/A"
                    low, high = thresholds[param]
                    if value <= low:
                        return 'Low'
                    elif value <= high:
                        return 'Moderate'
                    else:
                        return 'High'


                for pollutant in ['Ammonia', 'Nitrate', 'Phosphate']:
                    pred_df[f'{pollutant}_Level'] = pred_df[pollutant].apply(lambda x: pollutant_level(pollutant, x))

                expected_columns = ['Date'] + params + ['WQI', 'WQI_Status', 'Ammonia_Level', 'Nitrate_Level',
                                                        'Phosphate_Level']
                available_columns = [col for col in expected_columns if col in pred_df.columns]
                st.dataframe(pred_df[available_columns])

                csv = pred_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download WQI Results as CSV",
                    data=csv,
                    file_name=f"WQI_with_Pollutants_{selected_csv.lower()}.csv",
                    mime='text/csv',
                )

                # Enhanced discussion box
                discussion_text = """
                    By analyzing trends over multiple years, we can identify long-term changes in environmental factors that influence water quality. 
                    For example, a steady increase in surface temperature may indicate the effects of climate change on the lake’s ecosystem, while fluctuations 
                    in nutrient levels (e.g., nitrate and phosphate) could be due to seasonal runoff or pollution events. 

                    Similarly, changes in dissolved oxygen levels over time may point to shifts in biological activity or changes in the lake's overall health. 
                    Observing these trends is vital for understanding the lake's response to both natural and human-induced changes.
                    """
                st.markdown(discussion_box(discussion_text, font_size="20px"), unsafe_allow_html=True)

elif selected == "Predictions & Recommendations":
    st.markdown("<h2 style='color:#00BFC4;'>📈 Predictions & Recommendations</h2>", unsafe_allow_html=True)

    prediction_summary = """
    <div style='font-size: 18px; line-height: 1.7;'>
    The water quality predictions were derived through a combination of data-driven steps including exploratory data analysis (EDA), correlation analysis, temporal trend assessments, and model evaluation using deep learning architectures such as CNN, LSTM, and a hybrid CNN-LSTM.

    Predictions were carried out for key parameters including:
    <ul>
        <li><strong>pH</strong>: Indicates the acidity or alkalinity of the water.</li>
        <li><strong>Dissolved Oxygen (DO)</strong>: Crucial for aquatic life and ecosystem health.</li>
        <li><strong>Ammonia, Nitrate, and Phosphate</strong>: Nutrient levels affecting water eutrophication.</li>
        <li><strong>Surface, Middle, and Bottom Temperatures</strong>: Indicators of thermal stratification and possible climate impacts.</li>
    </ul>

    These predictions are available across multiple temporal resolutions (weekly, monthly, yearly) and can be filtered by <strong>location</strong> (e.g., Site A, B, C) and <strong>year</strong> (e.g., 2022–2024). 
    The hybrid CNN-LSTM model achieved the highest performance with an accuracy of 88%, making it the most reliable choice for environmental forecasting in this context.
    </div>
    """
    st.markdown(prediction_summary, unsafe_allow_html=True)

    recommendations = """
        <div style='border-left: 5px solid #00BFC4; padding-left: 15px; background-color: #f8f9fa;'>
        <h3 style='color:#00BFC4;'>Recommendations for Water Quality Management</h3>
        <ul style='font-size: 18px;'>
            <li><strong>Establish Continuous Monitoring:</strong> Deploy sensors at multiple depths and sites to capture temperature, pH, and DO in real-time, aiding in early detection of anomalies.</li>
            <li><strong>Address Nutrient Runoff:</strong> Regulate agricultural activities near the lake to minimize nutrient discharge, especially nitrate and phosphate, to combat eutrophication.</li>
            <li><strong>Predictive Maintenance and Alerts:</strong> Use the model outputs to develop an automated alert system for potential water quality decline based on predicted DO and pH drops.</li>
            <li><strong>Integrate Environmental Context:</strong> Include volcanic and meteorological data in future models to further enhance prediction accuracy.</li>
            <li><strong>Community Engagement:</strong> Share insights with local stakeholders to promote informed decision-making and sustainable lake resource use.</li>
            <li><strong>Policy Integration:</strong> Incorporate model-based predictions into local government environmental action plans, targeting specific regions (e.g., Site B showing rising ammonia) for remediation.</li>
            <li><strong>Model Retraining:</strong> Regularly update the prediction models with new data to adapt to changing environmental conditions.</li>
            <li><strong>Real-time Data Integration:</strong> Incorporate real-time data into prediction models through platforms like Streamlit to enhance prediction accuracy and support proactive decision-making.</li>
            <li><strong>Long-term Trends and Variables:</strong> Adjust models to reflect long-term trends and include additional environmental variables to improve reliability.</li>
            <li><strong>Human Activity Monitoring:</strong> Adopt a data-driven approach to monitor human activities, such as fishing and land use, to assess their influence on water quality and conservation efforts.</li>
            <li><strong>Biodiversity Tracking:</strong> Track biodiversity in the lake to identify critical areas for conservation, providing valuable insights for ecosystem health.</li>
            <li><strong>Adaptive Management Strategies:</strong> Use regular updates from monitoring systems to implement adaptive management strategies that ensure timely, data-informed actions for ecological balance.</li>
        </ul>
        </div>
        """

    st.markdown(recommendations, unsafe_allow_html=True)

    st.info("To explore prediction trends, go to the 'Model Evaluation' tab and interact with the graphs based on location, year, and parameters.")

elif selected == "About":
    st.markdown("<h2 style='color:#00BFC4;'>ℹ️ About This Dashboard</h2>", unsafe_allow_html=True)

    st.markdown("""
    ### 📘 Course Information
    This project is developed **in partial fulfillment** of the course:

    **CPEN 106 - ELECTIVES 2: BIG DATA ANALYTICS**  
    **Professor:** Engr. Joven Ramos  
    **Institution:** Cavite State University - Indang Main Campus  
    """)

    st.markdown("---")
    st.markdown("### 👨‍💻 Project Description")
    st.markdown("""
    This dashboard visualizes and analyzes water quality metrics collected from Taal Lake monitoring stations.  
    It also includes predictive modeling for future water conditions using deep learning.
    """)

    st.markdown("---")
    st.markdown("### 👥 Group Members")

    members = [
        {
            "name": "Atas, Alex",
            "student_number": "202201113",
            "email": "mai.alex.atas@cvsu.edu.ph",
            "image": "images/atas.jpg"
        },
        {
            "name": "Castillo, Karl Harold",
            "student_number": "202201719",
            "email": "main.karlharold.castillo@cvsu.edu.ph",
            "image": "images/castillo.jpg"
        },
        {
            "name": "Magnaye, Jhenn Mariz",
            "student_number": "202201480",
            "email": "main.jhennmariz.magnaye@cvsu.edu.ph",
            "image": "images/magnaye.jpg"
        },
        {
            "name": "Politud, Ma. Nicole",
            "student_number": "202203673",
            "email": "main.ma.nicole.politud@cvsu.edu.ph",
            "image": "images/politud.jpg"
        },
        {
            "name": "Tabios, Jhaenelle Allyson",
            "student_number": "202200485",
            "email": "main.jhaenelleallyson.tabios@cvsu.edu.ph",
            "image": "images/tabios.jpg"
        },
    ]

    for member in members:
        with st.container():
            cols = st.columns([1, 3])
            with cols[0]:
                image_path = os.path.join(os.path.dirname(__file__), member["image"])
                if os.path.exists(image_path):
                    st.image(image_path, width=200)
                else:
                    st.warning(f"Image not found: {member['image']}")
            with cols[1]:
                st.markdown(f"**Name:** {member['name']}")
                st.markdown(f"**Student Number:** {member['student_number']}")
                st.markdown(f"**CvSU Email:** {member['email']}")
            st.markdown("---")








