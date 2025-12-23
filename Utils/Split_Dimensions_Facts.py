import os
import pandas as pd

# Load the dataset (make sure this CSV is in your working directory first)
df = pd.read_csv("C:/AI Projects for Learners/Healthcare/Hospital_Bed_Occupancy_Wait_Time_Forecasting/Data/Data_Model/Hospital_Bed_Occupancy.csv")

# Define dimensions and fact tables mapping
dimensions = {
    "Patient": ["Patient_ID", "Age", "Gender", "Comorbidities_Count", "Insurance_Type"],
    "Admission": ["Admission_Type", "Primary_Diagnosis", "Severity_Score", "Arrival_Mode",
                  "Department", "Bed_Type", "Special_Requests", "Referral_Source"],
    "Hospital": ["Hospital_ID", "Total_Beds_Available", "ICU_Beds_Available",
                 "Staff_On_Duty", "Shift", "Average_Discharge_Rate_per_Hour", "Equipment_Availability"],
    "Temporal_Factors": ["Admission_Timestamp", "Discharge_Timestamp", "Day_of_Week",
                         "Season_Month", "Holiday_Flag", "Local_Event", "Weather_Conditions"],
    "Fact": ["Patient_ID", "Hospital_ID", "Wait_Time_to_Bed(Hours)", "ICU_Occupancy_Rate%",
             "ER_Crowding_Level", "Length_of_Stay(Hours)", "Discharge_Status"]
}

# Define base folder structure
base_path = r"C:\AI Projects for Learners\Healthcare\Hospital_Bed_Occupancy_Wait_Time_Forecasting\Data"
dimensions_folder = os.path.join(base_path, "Dimensions_Entities")
fact_folder = os.path.join(base_path, "Fact")

# Create folders if not exist
os.makedirs(dimensions_folder, exist_ok=True)
os.makedirs(fact_folder, exist_ok=True)

# Split and save CSVs
for dim, fields in dimensions.items():
    df_split = df[fields].copy()
    if dim == "Fact":
        file_path = os.path.join(fact_folder, f"{dim.lower()}_table.csv")
    else:
        file_path = os.path.join(dimensions_folder, f"{dim.lower()}_table.csv")
    df_split.to_csv(file_path, index=False)

print("✅ Dimension and Fact tables saved successfully!")
