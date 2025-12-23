import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import os
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost.callback import EarlyStopping
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR    
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

st.set_page_config(page_title="🏨 `Hospital Bed Occupancy & Wait Time Forecasting Application", layout="wide")

# Application Theme

st.markdown("""
    <style>
    body {
        #background-color: #84eab3;
        color: #212529;
    }

    .stApp {
        background-color: #47e8f5;
        font-family: 'Segoe UI', sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    header, footer {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)


# Custom CSS to apply background
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("C:/Users/Prasad/Desktop/Zenthic AI/Human Resources/HR360/HR_BG.png");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* Optional: Make expander header bigger and bold */
    details summary {
        font-size: 1.2rem !important;
        font-weight: 700;
    }

    /* Optional: Add a subtle light background to the expander content */
    .streamlit-expanderContent {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 1rem;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <style>
    .title-box {
        background-color: #62daa9;
        padding: 30px;
        border: 2px solid #6c757d;
        border-radius: 10px;
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: #000000;
        margin-bottom: 30px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .stRadio > div {
        flex-direction: row !important;
    }
    </style>
    <div class="title-box">
        🏨 Hospital Bed Occupancy & Wait Time Forecasting (MLops Solution)
    </div>
""", unsafe_allow_html=True)

def render_styled_table(df):
    st.markdown("""
        <style>
            .styled-table {
                border-collapse: collapse;
                margin: 25px 0;
                font-size: 0.9em;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                width: 100%;
                border: 1px solid #ddd;
            }
            .styled-table th, .styled-table td {
                padding: 12px 15px;
                border: 1px solid #ddd;
                text-align: center;
            }
            .styled-table thead {
                background-color: #009879;
                color: #ffffff;
            }
            .styled-table tbody tr:nth-child(even) {
                background-color: #f3f3f3;
            }
        </style>
    """, unsafe_allow_html=True)

    html_table = df.to_html(index=False, classes="styled-table")
    st.markdown(html_table, unsafe_allow_html=True)

st.markdown(" ")

st.markdown("""
<style>
.Section-box {
    background-color: #eeeeee;
    border-left: 6px solid #4CAF50;
    padding: 20px;
    margin: 15px 0;
    border-radius: 10px;
    font-family: 'Segoe UI', sans-serif;
}
.Section-title {
    font-size: 22px;
    font-weight: bold;
    color: #1a1a1a;
    margin-bottom: 10px;
}
.Section-text {
    font-size: 16px;
    color: #333333;
    line-height: 1.6;
}
.Section-text ul {
    margin: 10px 0 0 20px;
}
.Section-text li {
    margin-bottom: 6px;
}
</style>

<div class="Section-box">
    <div class="Section-title">🛠️ Hospital Bed Occupancy & Wait-Time Forecasting</div>
    <div class="Section-text">
        Managing hospital bed allocation and patient flow is a critical challenge in healthcare.  
        This project leverages AI-driven forecasting to predict **bed occupancy levels** and **patient wait-times**, 
        enabling hospitals to optimize resource allocation, reduce bottlenecks, and improve patient outcomes.
        <br><br>
        Using historical data, real-time inputs, and predictive analytics, this solution empowers healthcare providers with 
        proactive decision-making support.
        <ul>
            <li>📊 Forecast ICU, ER, and General Ward bed occupancy</li>
            <li>🕒 Predict patient wait-times for admission and bed allocation</li>
            <li>👩‍⚕️ Optimize staff and equipment utilization across shifts</li>
            <li>🌦️ Factor in seasonal trends, holidays, and local events impacting patient inflow</li>
        </ul>
        This forecasting engine ensures better patient care, efficient operations, and data-driven hospital management.
    </div>
</div>
""", unsafe_allow_html=True)

# File Path
file_path = r"C:\AI Projects for Learners\Healthcare\Hospital_Bed_Occupancy_Wait_Time_Forecasting\Data\Data_Model\Hospital_Bed_Occupancy.csv"

st.markdown(" ")
st.markdown(" ")

def load_data():
    df = pd.read_csv(file_path)

    # ✅ Convert Admission_Timestamp column to datetime
    df["Admission_Timestamp"] = pd.to_datetime(df["Admission_Timestamp"], errors="coerce")
    
    df["month"] = df["Admission_Timestamp"].dt.to_period("M").astype(str)
    df["year"] = df["Admission_Timestamp"].dt.year
    return df

st.subheader("📂 Step 1: Load Dataset")

df = load_data()

st.markdown(" ")

st.markdown(
    """
    <div style='    
        background-color: blue;
        padding: 10px 20px;
        border-radius: 10px;
        border: 1px solid #ccc;
        margin-bottom: 30px;
        text-align: center;
    '>
        <h4 style='color: white;'> Raw Data Preview </h4>
    """,
    unsafe_allow_html=True
)

render_styled_table(df.iloc[0:5,0:10])

st.markdown(" ")

with st.expander("📂 What is this section doing?", expanded=True):
    st.markdown("""
    - Loads the dataset into memory  
    - Displays the **first 5 rows** and **10 columns** for quick inspection  
    - Helps verify whether data is properly imported before moving to analysis  
    - Ensures data quality and structure are clear at the start  
    """, unsafe_allow_html=True)

st.markdown(" ")

with st.expander("🚀 Why is this significant?", expanded=True):
    st.markdown("""
    - Provides a **first glance** into the dataset structure  
    - Helps detect **missing values or anomalies early**  
    - Builds confidence that the **data pipeline is working** correctly  
    - Serves as the **foundation** for deeper analysis in the next steps  
    """, unsafe_allow_html=True)

st.markdown("---")

# ------------------------
# Data Cleaning
# ------------------------
st.subheader("🧹 Step 2: Data Cleaning")

st.markdown(" ")

df_cleaned = df.copy()

# Remove duplicates
df_cleaned = df_cleaned.drop_duplicates()

# Handle missing values (basic strategy: fill numeric with median, categorical with mode)
for col in df_cleaned.columns:
    if df_cleaned[col].dtype in ['int64', 'float64']:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    else:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])

# Convert Timestamps if they exist
for col in df_cleaned.columns:
    if "Timestamp" in col or "Date" in col:
        try:
            df_cleaned[col] = pd.to_datetime(df_cleaned[col])
        except:
            pass

#st.write("### Cleaned Dataset Preview", df_cleaned.head(10))

st.markdown(
    """
    <div style='
        background-color: blue;
        padding: 10px 20px;
        border-radius: 10px;
        border: 1px solid #ccc;
        margin-bottom: 30px;
        text-align: center;
    '>
        <h4 style='color: white;'> Cleaned Dataset Preview </h4>
    """,
    unsafe_allow_html=True
)

render_styled_table(df_cleaned.iloc[0:5,0:10])

st.markdown(" ")

with st.expander("📂 What is this step doing?", expanded=True):
    st.markdown("""
    This step ensures the dataset is **cleaned and standardized** before any modeling or analysis.  

    - Removes <strong>duplicate records</strong>  
    - Handles <strong>missing values</strong> by filling with <em>median (numeric)</em> or <em>mode (categorical)</em>  
    - Converts <strong>date/time columns</strong> into proper timestamp format  
    - Produces a **refined dataset** for accurate modeling  
    """, unsafe_allow_html=True)

with st.expander("📊 Why is this important?", expanded=True):
    st.markdown("""
    Clean data is essential for **reliable insights** and **robust AI/ML performance**.  

    - Prevents **bias and errors** from duplicate or missing entries  
    - Ensures models learn from **consistent, high-quality data**  
    - Improves **forecast accuracy** and **business decision-making**  
    - Saves significant **time in downstream analysis**  
    """, unsafe_allow_html=True)

st.markdown("---")

# ------------------------
# Summary Report
# ------------------------
st.subheader("📊 Step 3: Summary Report & Basic Insights")

st.markdown(" ")

# Shape
st.info(f"**Shape of Dataset:** {df_cleaned.shape[0]} rows × {df_cleaned.shape[1]} columns")

st.markdown(" ")
st.markdown(" ")

# Missing values
missing_report = df.isnull().sum().reset_index()
missing_report.columns = ["Column", "Missing Values"]
#st.write("**Missing Values Report**", missing_report)

st.markdown(
    """
    <div style='
        background-color: blue;
        padding: 10px 20px;
        border-radius: 10px;
        border: 1px solid #ccc;
        margin-bottom: 30px;
        text-align: center;
    '>
        <h4 style='color: white;'> Missing Values Report </h4>
    """,
    unsafe_allow_html=True
)

render_styled_table(missing_report.iloc[0:5,0:10])

st.markdown("---")

# Descriptive Statistics
descriptive_report = df_cleaned.describe(include='all').transpose()
#st.write("**Descriptive Statistics**", df_cleaned.describe(include='all').transpose())

st.markdown(
    """
    <div style='
        background-color: blue;
        padding: 10px 20px;
        border-radius: 10px;
        border: 1px solid #ccc;
        margin-bottom: 30px;
        text-align: center;
    '>
        <h4 style='color: white;'> Descriptive Statistics </h4>
    """,
    unsafe_allow_html=True
)

render_styled_table(descriptive_report.iloc[0:5,0:10])

st.markdown("---")

# Value counts for key categorical variables
categorical_cols = ["Admission_Type", "Department", "Bed_Type", "Discharge_Status"] 
#for col in categorical_cols:
    #if col in df_cleaned.columns:
        #st.write(f"**{col} Distribution**")
        #st.bar_chart(df_cleaned[col].value_counts())

col1, col2 = st.columns(2)

# --- Admission Type Distribution ---
with col1:
    st.markdown(
        """
        <div style='
            background-color: blue;
            padding: 10px 20px;
            border-radius: 10px;
            border: 1px solid #ccc;
            margin-bottom: 30px;
            text-align: center;
        '>
            <h4 style='color: white;'> Admission Type Distribution </h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    admission_counts = df_cleaned["Admission_Type"].value_counts().reset_index()
    admission_counts.columns = ["Admission_Type", "Count"]

    fig1 = px.bar(
        admission_counts,
        x="Admission_Type",
        y="Count",
        color_discrete_sequence=["blue"]
    )

    fig1.update_layout(
        plot_bgcolor="#47e8f5",
        paper_bgcolor="#47e8f5",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        font=dict(color="black", size=16)
    )

    st.plotly_chart(fig1, use_container_width=True)


# --- Department Distribution ---
with col2:
    st.markdown(
        """
        <div style='
            background-color: blue;
            padding: 10px 20px;
            border-radius: 10px;
            border: 1px solid #ccc;
            margin-bottom: 30px;
            text-align: center;
        '>
            <h4 style='color: white;'> Department Distribution </h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    dept_counts = df_cleaned["Department"].value_counts().reset_index()
    dept_counts.columns = ["Department", "Count"]

    fig2 = px.bar(
        dept_counts,
        x="Department",
        y="Count",
        color_discrete_sequence=["green"]
    )

    fig2.update_layout(
        plot_bgcolor="#47e8f5",
        paper_bgcolor="#47e8f5",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        font=dict(color="black", size=16)
    )

    st.plotly_chart(fig2, use_container_width=True)

col1, col2 = st.columns(2)

# --- Bed Type Distribution ---
with col1:
    st.markdown(
        """
        <div style='
            background-color: blue;
            padding: 10px 20px;
            border-radius: 10px;
            border: 1px solid #ccc;
            margin-bottom: 30px;
            text-align: center;
        '>
            <h4 style='color: white;'> Bed Type Distribution </h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    Bed_type_counts = df_cleaned["Bed_Type"].value_counts().reset_index()
    Bed_type_counts.columns = ["Bed_Type", "Count"]

    fig1 = px.bar(
        Bed_type_counts,
        x="Bed_Type",
        y="Count",
        color_discrete_sequence=["green"]
    )

    fig1.update_layout(
        plot_bgcolor="#47e8f5",
        paper_bgcolor="#47e8f5",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        font=dict(color="black", size=16)
    )

    st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

# --- Department Distribution ---
with col2:
    st.markdown(
        """
        <div style='
            background-color: blue;
            padding: 10px 20px;
            border-radius: 10px;
            border: 1px solid #ccc;
            margin-bottom: 30px;
            text-align: center;
        '>
            <h4 style='color: white;'> Discharge Staus Distribution </h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    Dischare_status_counts = df_cleaned["Discharge_Status"].value_counts().reset_index()
    Dischare_status_counts.columns = ["Discharge_Status", "Count"]

    fig2 = px.bar(
        Dischare_status_counts,
        x="Discharge_Status",
        y="Count",
        color_discrete_sequence=["red"]
    )

    fig2.update_layout(
        plot_bgcolor="#47e8f5",
        paper_bgcolor="#47e8f5",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        font=dict(color="black", size=16)
    )

    st.plotly_chart(fig2, use_container_width=True)

with st.expander("📂 What is this step doing?", expanded=True):
    st.markdown("""
    This step generates a **summary report** and uncovers **basic insights** about the dataset.  

    - Reports the <strong>shape</strong> of the dataset (rows × columns)  
    - Identifies <strong>missing values</strong> in each column  
    - Provides <strong>descriptive statistics</strong> (mean, median, std, min, max)  
    - Shows <strong>distribution of key categorical variables</strong> such as Admission Type, Department, Bed Type, and Discharge Status  
    - Presents results visually through <strong>bar charts</strong> and tables  
    """, unsafe_allow_html=True)

st.markdown(" ")
st.markdown(" ")

with st.expander("🚀 Why is this significant?", expanded=True):
    st.markdown("""
    This analysis is critical because it gives an **initial overview** of the dataset’s structure and potential data quality issues.  

    - Highlights **data gaps** and **irregularities** before deeper modeling  
    - Descriptive statistics help in **spotting anomalies** (e.g., very high wait times)  
    - Category distributions reveal **operational patterns** like:  
      - Which departments receive the most admissions  
      - Which bed types are most utilized  
      - Common discharge outcomes  
    - Sets the stage for more advanced **exploratory data analysis (EDA)** and **predictive modeling**  
    """, unsafe_allow_html=True)

st.markdown("---")

# ------------------------
# Initial Insights
# ------------------------

st.markdown(
    """
    <div style='
        background-color: blue;
        padding: 10px 20px;
        border-radius: 10px;
        border: 1px solid #ccc;
        margin-bottom: 30px;
        text-align: center;
    '>
        <h4 style='color: white;'> Basic Insights </h4>
    """,
    unsafe_allow_html=True
)

st.markdown(" ")

#st.subheader("🔎 Basic Insights")


col1, col2, col3 = st.columns(3)

#with col1:
    #if "Wait_Time_to_Bed(Hours)" in df_cleaned.columns:
        #st.metric("Average Wait Time to Bed (mins)", round(df_cleaned["Wait_Time_to_Bed(Hours)"].mean(), 1))

with col1:
    if "Wait_Time_to_Bed(Hours)" in df_cleaned.columns:
        avg_wait = round(df_cleaned["Wait_Time_to_Bed(Hours)"].mean() * 60, 1)  # converting hours → minutes
        st.markdown(
            f"""
            <div style="
                background-color: green;
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
                color: #1a1a1a;
                font-family: 'Arial', sans-serif;
            ">
                <h4 style="margin: 0; font-size: 18px; color: white;">⏱️ Average Wait Time</h4>
                <p style="margin: 5px 0; font-size: 28px; font-weight: bold; color: white;">
                    {avg_wait} mins
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

#with col2:
    #if "Length_of_Stay(Hours)" in df_cleaned.columns:
        #st.metric("Average Length of Stay (hrs)", round(df_cleaned["Length_of_Stay(Hours)"].mean(), 1))

with col2:
    if "Length_of_Stay(Hours)" in df_cleaned.columns:
        avg_stay = round(df_cleaned["Length_of_Stay(Hours)"].mean(), 1)
        st.markdown(
            f"""
            <div style="
                background-color: green;
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
                color: #1a1a1a;
                font-family: 'Arial', sans-serif;
            ">
                <h4 style="margin: 0; font-size: 18px; color: white;">🏥 Avg Length of Stay</h4>
                <p style="margin: 5px 0; font-size: 28px; font-weight: bold; color: white;">
                    {avg_stay} hrs
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

#with col3:
    #if "ICU_Occupancy_Rate%" in df_cleaned.columns:
        #st.metric("Average ICU Occupancy Rate (hrs)", round(df_cleaned["ICU_Occupancy_Rate%"].mean(), 1))

with col3:
    if "ICU_Occupancy_Rate%" in df_cleaned.columns:
        avg_icu = round(df_cleaned["ICU_Occupancy_Rate%"].mean(), 1)
        st.markdown(
            f"""
            <div style="
                background-color: green;
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
                color: #1a1a1a;
                font-family: 'Arial', sans-serif;
            ">
                <h4 style="margin: 0; font-size: 18px; color: white;">🛏️ Avg ICU Occupancy</h4>
                <p style="margin: 5px 0; font-size: 28px; font-weight: bold; color: white;">
                    {avg_icu} %
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("---")

if "Admission_Type" in df_cleaned.columns:
    top_admission = df_cleaned["Admission_Type"].value_counts().idxmax()
    st.success(f"✅ Most common Admission Type: **{top_admission}**")

if "Department" in df_cleaned.columns:
    busiest_dept = df_cleaned["Department"].value_counts().idxmax()
    st.success(f"🏥 Busiest Department: **{busiest_dept}**")

st.markdown(" ")

with st.expander("📂 What is this step doing?", expanded=True):
    st.markdown("""
    This step highlights some **quick operational insights** using simple metrics and summaries.  

    - Calculates **average wait time** (in minutes) before a patient gets a bed  
    - Reports **average length of stay (hours)** across patients  
    - Computes **average ICU occupancy rate (%)**  
    - Identifies the **most common admission type**  
    - Highlights the **busiest department** in terms of patient flow  

    These insights are presented using **stylized info-cards** and **highlighted messages** for clarity.  
    """, unsafe_allow_html=True)

st.markdown(" ")
st.markdown(" ")

with st.expander("🚀 Why is this significant?", expanded=True):
    st.markdown("""
    These quick insights provide **immediate visibility** into key hospital performance indicators.  

    - Average wait times indicate **efficiency in patient allocation**  
    - Length of stay reflects **treatment duration & hospital turnover**  
    - ICU occupancy highlights **critical care load**  
    - Admission type distribution shows **how patients enter the system** (e.g., emergency vs scheduled)  
    - Department-level insights help in **capacity planning** and **resource allocation**  

    Together, these metrics form a **high-level dashboard** that can guide hospital administrators in decision-making and spotting operational bottlenecks early.  
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("---")

st.markdown(
    f"""
    <div style="
        background-color: Blue;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
        color: #1a1a1a;
        font-family: 'Arial', sans-serif;
    ">
        <h4 style="margin: 0; font-size: 24px; color: white;">Exploratory Data Analysis</h4>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(" ")

st.markdown("""
<div class="Section-box">
    <div class="Section-title">🔎 Exploratory Data Analysis (EDA)</div>
    <div class="Section-text">
        Exploratory Data Analysis (EDA) is the process of visually and statistically exploring datasets 
        to uncover patterns, detect anomalies, test assumptions, and validate data quality before applying predictive models.  
        <br><br>
        In the context of **Hospital Bed Occupancy & Wait-Time Forecasting**, EDA helps us:
        <ul>
            <li>📈 Understand patient admission trends, discharge patterns, and seasonal variations</li>
            <li>🔍 Identify correlations between occupancy, wait-times, and length of stay</li>
            <li>⚠️ Detect missing values, outliers, and inconsistencies in hospital records</li>
            <li>📊 Reveal capacity bottlenecks in ICU, ER, and General wards</li>
        </ul>
        <br>
        <b>Benefits of EDA in Healthcare Forecasting:</b>
        <ul>
            <li>🧭 Builds a clear understanding of hospital operations before forecasting</li>
            <li>💡 Provides data-driven insights for resource planning and policy-making</li>
            <li>🤝 Enhances collaboration between data scientists, doctors, and administrators</li>
            <li>🚀 Increases model accuracy by ensuring data quality and relevance</li>
        </ul>
        By conducting EDA, we ensure that our predictive modeling is grounded in reliable data, 
        leading to better forecasts, optimized resource allocation, and improved patient outcomes.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown(" ")

col1, col2 = st.columns(2)

# 📈 Objective 1: Admission Trends, Discharge Patterns, Seasonality

with col1:
    st.markdown(
        """
        <div style='
            background-color: blue;
            padding: 10px 20px;
            border-radius: 10px;
            border: 1px solid #ccc;
            margin-bottom: 30px;
            text-align: center;
        '>
            <h4 style='color: white;'> 📈 Patient Admission & Discharge Trends </h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    admissions = df.groupby("month").size().reset_index(name="admissions")
    discharges = df.groupby("month")["Discharge_Timestamp"].count().reset_index(name="discharges")
    merged = pd.merge(admissions, discharges, on="month")

    # Line chart
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=merged["month"], y=merged["admissions"],
        mode='lines+markers', name='Admissions',
        line=dict(color="#FF6F61", width=3),  # custom color
        marker=dict(size=6)
    ))
    fig1.add_trace(go.Scatter(
        x=merged["month"], y=merged["discharges"],
        mode='lines+markers', name='Discharges',
        line=dict(color="#6A5ACD", width=3),  # another custom color
        marker=dict(size=6)
    ))

    fig1.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",  # transparent
        paper_bgcolor="rgba(0,0,0,0)",  # transparent
        xaxis_title="Month",
        yaxis_title="Count",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        font=dict(color="black", size=16)
    )
    st.plotly_chart(fig1, use_container_width=True)

# 🔍 Objective 2: Correlation Analysis

with col2:
    st.markdown(
        """
        <div style='
            background-color: blue;
            padding: 10px 20px;
            border-radius: 10px;
            border: 1px solid #ccc;
            margin-bottom: 30px;
            text-align: center;
        '>
            <h4 style='color: white;'> 🔍 Correlation Between Occupancy, Wait-Time & LOS </h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    corr_matrix = df[["ICU_Occupancy_Rate%", "Wait_Time_to_Bed(Hours)", "Length_of_Stay(Hours)"]].corr()

    fig2 = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Plasma"  # nicer color scale
    )

    fig2.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="black", size=14)
    )

    st.plotly_chart(fig2, use_container_width=True)

st.markdown(" ")

with st.expander("📂 What is this step doing?", expanded=True):
    st.markdown("""
    In this step, we are conducting **Exploratory Data Analysis (EDA)** to better understand our dataset.  

    Specifically, this includes:  
    - 📈 Analyzing **patient admission & discharge trends** across months (to detect seasonality and demand surges)  
    - 🔍 Computing **correlations** between ICU occupancy, wait times, and length of stay  
    - ⚠️ Identifying anomalies, bottlenecks, and data quality issues  
    - 📊 Visualizing patterns in hospital operations before moving to predictive modeling  

    By doing this, we ensure that the data is **well understood** and **ready for forecasting models**.  
    """, unsafe_allow_html=True)

st.markdown(" ")
st.markdown(" ")

with st.expander("🚀 Why is this significant?", expanded=True):
    st.markdown("""
    EDA is **critical in healthcare analytics** because it uncovers the hidden patterns behind hospital operations.  

    - Trends in admissions and discharges help in **resource & staff planning**  
    - Correlation analysis shows how **wait time, ICU load, and stay duration** are interconnected  
    - Detecting anomalies ensures **data reliability** before applying machine learning models  
    - Understanding seasonality supports **demand forecasting** for peak vs. low patient load  

    ✅ In short, EDA provides the **foundation for accurate forecasting** and enables hospitals to plan resources, reduce wait times, and improve patient care outcomes.  
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(" ")

# Styled bounding box with background color
st.markdown(
    """
    <div style='
        background-color: blue;
        padding: 40px;
        border-radius: 10px;
        width: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #d1e8ff;
        margin-bottom: 30px;
    '>
    <h3 style='text-align: center; color: #FFFFFF;'>🧠 Machine Learning Implementation</h3>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style='background-color: green; padding: 10px 30px; border-radius: 12px; border: 1px solid #ccc; margin-top: 30px;'>
    <h2 style='text-align:center; color:white;'> 🔍 1. Wait Time to Bed Prediction with Linear Regression </h2>
    """,
    unsafe_allow_html=True
)

#st.subheader("🔍 1. Wait Time to Bed Prediction with Linear Regression")
st.markdown(" ", unsafe_allow_html=True)
st.markdown("---", unsafe_allow_html=True)

st.markdown(
    """
    <div style='background-color: #DDDDDD; padding: 10px 30px; border-radius: 12px; border: 2px solid #bbb; margin-top: 30px;'>
    <h3 style='text-align:center; color:#333;'>Feature Engineering </h3>
    """,
    unsafe_allow_html=True
)

st.markdown(" ", unsafe_allow_html=True)           
st.markdown(" ", unsafe_allow_html=True)    

st.markdown("""
<div class="Section-box">
    <div class="Section-title">⚙️ Feature Engineering (Extraction & Selection)</div>
    <div class="Section-text">
        Feature Engineering is the foundation of building robust Machine Learning models.  
        It involves extracting meaningful variables from raw data and selecting the most impactful features for predictions.  
        By transforming, encoding, and scaling data properly, we improve the model’s ability to **learn patterns** effectively.  
        <br><br>
        In this project, we apply:
        <ul>
            <li>📥 Feature Extraction – deriving new variables from raw data</li>
            <li>🎯 Feature Selection – choosing the most relevant predictors</li>
            <li>🔤 Encoding – converting categorical data into numeric form</li>
            <li>📏 Scaling – normalizing numerical values for fair comparison</li>
        </ul>
        In this step, we ensure data is cleaned, relevant attributes are created, and only the most predictive ones are used.  
        <ul>
            <li>📌 Handle missing values, outliers, and noisy data</li>
            <li>🔄 Encode categorical variables & normalize numeric features</li>
            <li>🧮 Create new features from existing data (domain-driven engineering)</li>
            <li>🎯 Select the best subset of features using statistical & ML-based methods</li>
        </ul>
        This step directly influences model accuracy, interpretability, and generalization performance.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(" ")
st.markdown(" ")

# -----------------
# Feature Selection Mode
# -----------------
st.markdown(
    """
    <div style='
        background-color: blue;
        padding: 10px 20px;
        border-radius: 10px;
        border: 1px solid #ccc;
        margin-bottom: 30px;
        text-align: center;
    '>
        <h4 style='color: white;'> 🎯 Choose Feature Selection Approach </h4>
    </div>
    """,
    unsafe_allow_html=True
)

#st.markdown("## 🎯 Choose Feature Selection Approach")

col1, col2, col3 = st.columns([1,1,1])  # middle column is wider

with col2:
    feature_selection_mode = st.radio(
         " ",
        ["Manual Selection", "Automated Selection"],
        index=0,
        horizontal=True
    )

st.markdown(" ")
st.markdown(" ")

if feature_selection_mode == "Automated Selection":

    # -----------------
    # Target Selection
    # -----------------

    st.markdown(
        """
        <div style='
            background-color: blue;
            padding: 10px 20px;
            border-radius: 10px;
            border: 1px solid #ccc;
            margin-bottom: 30px;
            text-align: left;
        '>
            <h4 style='color: white;'> 🎯 Select Target Variable </h4>
        </div>
        """,
        unsafe_allow_html=True
    )
    #st.subheader("🎯 Select Target Variable")
    st.markdown(" ")

    # Ensure the target column exists
    if "Wait_Time_to_Bed(Hours)" in df.columns:
        default_index = list(df.columns).index("Wait_Time_to_Bed(Hours)")
    else:
        default_index = 0  # fallback if column not found

    # Selectbox with default selection
    target = st.selectbox("Choose your target column:", df.columns, index=default_index)

    # Clean column names
    df.columns = df.columns.str.strip()       # remove leading/trailing spaces
    df.columns = df.columns.str.replace("\u200b", "", regex=False)  # remove hidden zero-width spaces
    df.columns = df.columns.str.replace(" ", "_")      # replace spaces with underscores
    st.markdown(" ")

    #render_styled_table(df.iloc[:5])

    # -----------------
    # Feature Selection Methods
    # -----------------

    st.markdown(" ")
    #st.subheader("⚡ Feature Selection Methods")

    #method = st.radio(" ", ["Correlation with Target", "SelectKBest", "Recursive Feature Elimination (RFE)", "RandomForest Feature Importance"])

    options = [
        "Correlation with Target",
        "SelectKBest",
        "Recursive Feature Elimination (RFE)",
        "RandomForest Feature Importance",
    ]

    with st.container(border=True):
        st.markdown("### ⚡ Feature Selection Methods")
        method = option_menu(
            None, options,
            icons=["bar-chart", "list-ol", "scissors", "tree"],
            orientation="horizontal",
            default_index=0,
            styles={
                "container": {"background-color": "#eeeeee", "border-radius": "0px", "border": "1px solid black", "padding": "12px"},
                "nav-link": {"font-size": "14px", "padding": "8px 14px",
                             "border-radius": "9999px", "margin": "4px"},
                "nav-link-selected": {"background-color": "darkblue", "color": "white"},
            },
        )

    # ----------------------------------
    # 1. Correlation Method
    # ----------------------------------

    if method == "Correlation with Target":
        st.subheader("📊 Showing correlation of numeric features with the target variable:")

        if target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
            df.drop(columns=['Wait_Time_to_Bed(Hours)'], errors='ignore')
            numeric_df = df.select_dtypes(include=['float64', 'int64'])
            corr = numeric_df.corr()[target].sort_values(ascending=False)

            # Store top correlated features (exclude target itself)
            features = corr.drop(target).head(5).index.tolist()
            st.session_state["selected_features"] = features

            st.success(f"✅ Selected features saved to session: {features}")

            # Plot correlation
            corr_df = corr.reset_index()
            corr_df.columns = ["Feature", "Correlation"]
            fig = px.bar(
                corr_df, x="Correlation", y="Feature",
                orientation="h", color="Correlation",
                color_continuous_scale="RdBu", text="Correlation",
                height=500, width=700,
            )
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")

            # 🔹 Transparent background
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning(f"⚠️ Target '{target}' is not numeric. Correlation skipped.")

    # -----------------
    # 2. SelectKBest
    # -----------------

    elif method == "SelectKBest":
        X = df.drop(columns=[target], errors="ignore")
        X = df.select_dtypes(include=['float64', 'int64'])
        y = df[target]

        k = st.slider("Select number of top features:", 1, len(X.columns), 4)
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X, y)

        features = X.columns[selector.get_support()].tolist()
        if target in features:
            features.remove(target)
        st.session_state["selected_features"] = features

        st.success(f"✅ SelectKBest Selected Features are : {features}")
        #st.success(f"✅ {len(features)} features saved to session: {features}")
        render_styled_table(pd.DataFrame({"Top Selected Features": features}))

    # -----------------
    # 3. Recursive Feature Elimination (RFE)
    # -----------------

    elif method == "Recursive Feature Elimination (RFE)":
        X = df.drop(columns=[target], errors="ignore")
        X = df.select_dtypes(include=['float64', 'int64'])
        y = df[target]

        model = LogisticRegression(max_iter=1000)
        rfe = RFE(model, n_features_to_select=5)
        rfe.fit(X, y)

        features = X.columns[rfe.get_support()].tolist()
        if target in features:
            features.remove(target)        
        st.session_state["selected_features"] = features

        st.success(f"✅ RFE Selected Features Saved are : {features}")
        render_styled_table(pd.DataFrame({"Top Selected Features": features}))

    # -----------------
    # 4. RandomForest Feature Importance
    # -----------------

    elif method == "RandomForest Feature Importance":
        X = df.drop(columns=[target], errors="ignore")
        X = df.select_dtypes(include=['float64', 'int64'])
        y = df[target]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        features = importances.head(5).index.tolist()
        if target in features:
            features.remove(target)
        st.session_state["selected_features"] = features

        st.success(f"✅ RandomForest Selected Features are : {features}")
        render_styled_table(pd.DataFrame({"Top Selected Features": features}))

    # -----------------
    # Correlation Heatmap
    # -----------------

    st.markdown("---")
    st.header("📊 Feature Correlation Heatmap")

    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        corr = numeric_df.corr()
        fig_width = min(12, max(6, 0.6 * len(corr.columns)))
        fig_height = fig_width * 0.8

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Transparent background
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")

        sns.heatmap(
            corr,
            cmap="coolwarm",
            annot=len(corr.columns) <= 10,
            fmt=".2f",
            linewidths=0.5,
            linecolor="white",
            cbar_kws={'shrink': 0.6},
            ax=ax
        )

        st.pyplot(fig, transparent=True)
    else:
        st.warning("⚠️ No numeric columns available for correlation analysis. Please apply encoding first.")

    st.markdown("---")


elif feature_selection_mode == "Manual Selection":

    # -----------------
    # Manual Feature Selection
    # -----------------

    features = [
        "Age", "Gender", "Admission_Type", "Primary_Diagnosis", "Comorbidities_Count",
        "Severity_Score", "Arrival_Mode", "Department", "Bed_Type", "Special_Requests",
        "Referral_Source", "Total_Beds_Available", "ICU_Beds_Available", "Staff_On_Duty",
        "Shift", "Average_Discharge_Rate_per_Hour", "Equipment_Availability", "Day_of_Week",
        "Season_Month", "Holiday_Flag", "Local_Event", "Weather_Conditions",
        "ICU_Occupancy_Rate%", "ER_Crowding_Level", "Length_of_Stay(Hours)"
    ]

    # Override with final manual selection
    features = ["Age", "Severity_Score"]

    st.info("The Manually Selected Features are: **Age & Severity Score**")

    target = "Wait_Time_to_Bed(Hours)"  # default target for manual mode

# Filter dataset with only required columns
df1 = df[features]

# Separate categorical and numeric features
categorical_cols = ["Gender", "Admission_Type", "Primary_Diagnosis", "Arrival_Mode", "Department",
    "Bed_Type", "Special_Requests", "Referral_Source", "Shift", "Equipment_Availability",
    "Day_of_Week", "Season_Month", "Holiday_Flag", "Local_Event", "Weather_Conditions", "ER_Crowding_Level"]

numeric_cols = [col for col in df1.columns if col not in categorical_cols]

# One-Hot Encode categorical columns
encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)  # 🔑 note sparse=False ensures dense numpy array instead of sparse matrix
encoded_array = encoder.fit_transform(df[categorical_cols])

# Convert encoded array back to DataFrame with column names
encoded_df = pd.DataFrame(
    encoded_array,
    columns=encoder.get_feature_names_out(categorical_cols),
    index=df.index
)

# Combine numeric + encoded categorical
df2 = pd.concat([df1[numeric_cols], encoded_df], axis=1)

df3 = df2.copy()

st.markdown(
    """
    <div style='background-color: #DDDDDD; padding: 10px 30px; border-radius: 12px; border: 2px solid #bbb; margin-top: 30px;'>
    <h3 style='text-align:center; color:#333;'> Model Engineering </h3>
    """,
    unsafe_allow_html=True
)

st.markdown(" ")

st.markdown("""
<div class="Section-box">
    <div class="Section-title">🤖 Model Engineering (Training, Evaluation & Testing)</div>
    <div class="Section-text">
        Model Engineering focuses on training Machine Learning algorithms, evaluating their performance, 
        and addressing underfitting/overfitting challenges to ensure **reliable predictions**.  
        This is where data meets algorithms to create a predictive engine.  
        <br><br>
        We experiment with multiple models, tune hyperparameters, and validate them on unseen data before final deployment.  
        <ul>
            <li>🧑‍🏫 Train models on engineered features using supervised/unsupervised algorithms</li>
            <li>📊 Evaluate performance with metrics like Accuracy, RMSE, Precision, Recall, F1-score</li>
            <li>⚖️ Address underfitting (bias) & overfitting (variance) using cross-validation & regularization</li>
            <li>🧪 Test final model on hold-out data to measure real-world performance</li>
        </ul>
        This step ensures the AI system is not only accurate but also robust and generalizable to future data.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(" ")
st.markdown(" ")

with st.container(border=True):
    st.markdown("<h3 style='text-align:center;'>🎯 Select Target Variable</h3>", unsafe_allow_html=True)
    
    if "Wait_Time_to_Bed(Hours)" in df.columns:
        default_index = list(df.columns).index("Wait_Time_to_Bed(Hours)")
    
    else:
        default_index = 0
    target_col = st.selectbox("Select your Target Variable:", df.columns, index=default_index)

    st.markdown("</div>", unsafe_allow_html=True)

X = df3    
y = df[target_col].copy()

with st.container(border=True):
    # Open styled box
    st.markdown(
        """
        <div style="
            background-color: #84eab3;
            border: 3px black;
            border-radius: 8px;
            color: black;
            padding: -4px;
            text-align: center
        ">
            <h3 style="margin-top:0;">  📐 Test Size (%)</h3>
        """,
        unsafe_allow_html=True
    )

    st.markdown(" ")
    st.markdown(" ")    

    # Place slider inside the styled div
    test_size = st.slider("Adjust the Test Data Size", 5, 50, 20, step=5) / 100

    # Close styled box
    st.markdown("</div>", unsafe_allow_html=True)

with st.container(border=True):
    st.markdown(
        """
        <div style="
            background-color: #84eab3;
            border: 3px black;
            border-radius: 8px;
            color: black;
            padding: -4px;
            text-align: center
        ">
            <h3 style="margin-top:0;">  🔑 Random State </h3>
        """,
        unsafe_allow_html=True
    )

    st.markdown(" ")
    st.markdown(" ")

    #st.markdown("### 🔑 Random State")    
    random_state = st.number_input("Initialize a Random State Valaue", min_value=0, max_value=999, value=42, step=1)

    # Train-test split (after features built)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    st.info(f"📊 Training Samples: {X_train.shape[0]} | Testing Samples: {X_test.shape[0]}")

    # Close styled box
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(" ")
st.markdown(" ")
st.markdown(" ")

# -----------------
# Model Setup & Target Selection
# -----------------
with st.container(border=True):
    st.markdown(
        "<h3 style='text-align:left; color:black; '>✅ Using XGBoost Regressor as the Default Model </h3>",
        unsafe_allow_html=True
    )
    #st.write("✅ Using XGBoost Regressor as the default model")    

# -----------------
# Initialize XGBoost with safe defaults
# -----------------

# Split training data for early stopping validation
X_tr_full, X_val, y_tr_full, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=random_state
)

# Scale data manually (instead of using pipeline for early stopping phase)

scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr_full)
X_val_scaled = scaler.transform(X_val)
X_full_scaled = scaler.fit_transform(X_train)


# Train XGB with early stopping on scaled data
xgb_model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.5,
    random_state=random_state,
    n_jobs=-1
)

# -----------------
# Train & Evaluate
# -----------------

xgb_model.fit(
    X_tr_scaled, y_tr_full,
    eval_set=[(X_val_scaled, y_val)],
    #early_stopping_rounds=50,
    verbose=True
)

# Get best iteration if available
best_n = getattr(xgb_model, "best_iteration", None)
if best_n is not None:
    best_n += 1
    st.write(f"✅ Best boosting round: {best_n}")

# Rebuild pipeline with the tuned model
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb", XGBRegressor(
        n_estimators=best_n if best_n else 500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.5,
        random_state=random_state,
        n_jobs=-1
    ))
])

# Final fit on full training data
pipe.fit(X_train, y_train)

st.info("✅ Model Training Done Successfully with Early Stopping")

# -----------------
# Test Evaluation
# -----------------

st.markdown(
    f"""
    <div style='background-color: green; padding: 10px 30px; border-radius: 12px; border: 2px solid #bbb; margin-top: 20px;'>
    <h4 style='text-align:center; color: white;'> Model Evaluation </h4>
    </div>
    """,
    unsafe_allow_html=True
)    

# -----------------
# Cross-Validation Evaluation
# -----------------

# Cross-validation using pipeline (safer)

cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
cv_mean, cv_std = np.mean(cv_scores), np.std(cv_scores)

# -----------------
# Predictions
# -----------------

y_train_pred = pipe.predict(X_train)
y_test_pred = pipe.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

results_df = pd.DataFrame([{
    "Model": "XGBoost Regressor (pipeline)",
    "Train R²": train_r2,
    "Test R²": test_r2,
    "RMSE": rmse,
    "CV R² (mean)": cv_mean,
    "CV R² (std)": cv_std
}]).set_index("Model")

# -----------------
# Diagnosis
# -----------------

# Diagnosis helper
def diagnose(train_r2, test_r2, tol=0.20):
    if train_r2 < 0.5 and test_r2 < 0.5:
        return "❌ Underfitting: add features or reduce regularization"
    elif train_r2 - test_r2 > tol:
        return "⚠️ Overfitting: try stronger regularization, early stopping, reduce complexity, add data"
    else:
        return "✅ Generalizing well"

results_df["Diagnosis"] = results_df.apply(lambda row: diagnose(row["Train R²"], row["Test R²"]), axis=1)

# -----------------
# Display Results
# -----------------

render_styled_table(results_df)

# -----------------

# Learning curve (use pipeline)

# -----------------

train_sizes, train_scores, test_scores = learning_curve(pipe, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

fig = go.Figure()
fig.add_trace(go.Scatter(x=train_sizes, y=train_mean, mode="lines+markers", name="Train R²"))
fig.add_trace(go.Scatter(x=train_sizes, y=test_mean, mode="lines+markers", name="Test R²"))
fig.update_layout(title="Learning Curve (XGBoost pipeline)", xaxis_title="Training Samples", yaxis_title="R²",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, key="learning_curve_xgb")

st.success(f"📌 Final Diagnosis: {diagnose(train_r2, test_r2)}")
st.info(f"Cross-Validation R²: {cv_mean:.3f} ± {cv_std:.3f}")

st.markdown(" ")

# -----------------
# Model Testing
# -----------------

st.markdown(
    f"""
    <div style='background-color: orange; padding: 10px 30px; border-radius: 12px; border: 2px solid #bbb; margin-top: 20px;'>
    <h4 style='text-align:center; color: white;'> Model Testing</h4>
    </div>
    """,
    unsafe_allow_html=True
)

# Example: Show sample predictions vs actuals
st.markdown(" ")

st.write("🔍 Sample Predictions vs Actuals:")
comparison_df = (
    pd.DataFrame({"Actual": y_test[:10], "Predicted": y_test_pred[:10]})
    .reset_index(drop=True)
)
render_styled_table(comparison_df)

st.markdown(" ")    
st.info("Model Testing Done Successfully")

# -----------------
# Bar Plot for Comparison (Interactive)
# -----------------

results = {
    #"Linear Regression": [mae_train, mae_test, rmse_train, rmse_test, r2_train, r2_test],
    "XGBoost Regressor": [rmse, train_r2, test_r2],
}

# Convert dict of lists → dataframe
results_df = pd.DataFrame.from_dict(
    results, 
    orient="index", 
    columns=["RMSE", "Train R²", "Test R²"]
).reset_index()

results_df.rename(columns={"index": "Model"}, inplace=True)

st.markdown(" ")
st.markdown(" ")

# -----------------
# Prediction vs Actual Plot (Test Set)
# -----------------

y_pred_best = pipe.predict(X_test)

base_model_name = results_df.loc[results_df["Test R²"].idxmax(), "Model"]

st.markdown(
    f"""
    <div style='
        background-color: blue;
        padding: 10px 20px;
        border-radius: 10px;
        border: 1px solid #ccc;
        margin-bottom: 30px;
        text-align: left;
    '>
        <h4 style='color: white;'> Actual vs Predicted (Test) - {base_model_name} </h4>
    </div>
    """,
    unsafe_allow_html=True
)
#st.subheader(f"Actual vs Predicted (Test) - {base_model_name}")

fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(
    x=y_test, y=y_test_pred, mode="markers",
    name="Predicted vs Actual", marker=dict(color="blue", opacity=0.6)
))
fig_scatter.add_trace(go.Scatter(
    x=[y_test.min(), y_test.max()],
    y=[y_test.min(), y_test.max()],
    mode="lines", name="Perfect Fit", line=dict(color="red", dash="dash")
))
fig_scatter.update_layout(
    #title=f"Actual vs Predicted (Test) - {base_model_name}",
    xaxis_title="Actual",
    yaxis_title="Predicted",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    height=600
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown(" ")
st.markdown(" ")

# -----------------
# Actual(y) vs Pred (y_pred) [All Records]
# -----------------
y_pred_all = pipe.predict(X)

base_model_name = results_df.loc[results_df["Test R²"].idxmax(), "Model"]

st.markdown(
    f"""
    <div style='
        background-color: blue;
        padding: 10px 20px;
        border-radius: 10px;
        border: 1px solid #ccc;
        margin-bottom: 30px;
        text-align: left;
    '>
        <h4 style='color: white;'> Actual vs Predicted (All Data) - {base_model_name} </h4>
    </div>
    """,
    unsafe_allow_html=True
)
#st.subheader(f"Actual vs Predicted (All Data) - {base_model_name}")

fig_all = go.Figure()
fig_all.add_trace(go.Scatter(
    x=y, y=y_pred_all, mode="markers",
    name="Predicted vs Actual", marker=dict(color="green", opacity=0.6)
))
fig_all.add_trace(go.Scatter(
    x=[y.min(), y.max()],
    y=[y.min(), y.max()],
    mode="lines", name="Perfect Fit", line=dict(color="red", dash="dash")
))
fig_all.update_layout(
    #title="Actual vs Predicted (All Data)",
    xaxis_title="Actual Values",
    yaxis_title="Predicted Values",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    height=600
)
st.plotly_chart(fig_all, use_container_width=True)

st.markdown(" ")
st.markdown(" ")

# -----------------
# Residuals Plot (Distribution)
# -----------------
base_model_name = results_df.loc[results_df["Test R²"].idxmax(), "Model"]

st.markdown( 
    f"""
    <div style='
        background-color: blue;
        padding: 10px 20px;
        border-radius: 10px;
        border: 1px solid #ccc;
        margin-bottom: 30px;
        text-align: left;
    '>
        <h4 style='color: white;'> Residuals Distribution - {base_model_name} </h4>
    </div>
    """,
    unsafe_allow_html=True
)
#st.subheader(f"Residuals Distribution - {base_model_name}")

residuals = y_test - y_pred_best
fig_resid = px.histogram(
    residuals, nbins=30, marginal="box", opacity=0.7
    #title=f"Residuals Distribution - {base_model_name}"
)
fig_resid.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    height=600,
    title_x=0.5
)
st.plotly_chart(fig_resid, use_container_width=True)

st.markdown(" ")
st.markdown(" ")