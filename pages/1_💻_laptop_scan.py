import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define the LaptopDataPreprocessor class once and use it throughout the code
class LaptopDataPreprocessor:
    def __init__(self):
        self.categorical_features = ['Laptop Model', 'Laptop Status', 'Manufacturer', 'Processor Type', 'Graphics Card', 'Bluetooth', 'Wi-Fi', 'Touch Screen']
        self.numerical_features = ['Disk Usage (%)', 'CPU Usage (%)', 'Memory Usage (%)', 'Screen Size (inch)', 'Battery Capacity (Wh)', 'Number of USB Ports', 'Weight (kg)']
        
        self.preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(drop='first'), self.categorical_features),
            ('num', StandardScaler(), self.numerical_features)
        ])

    def transform(self, df):
        X_preprocessed = self.preprocessor.transform(df)
        return X_preprocessed

    def fit_transform(self, df):
        X_preprocessed = self.preprocessor.fit_transform(df)
        return X_preprocessed

    def save(self, file_path):
        joblib.dump(self.preprocessor, file_path)

    @staticmethod
    def load(file_path):
        preprocessor = joblib.load(file_path)
        return preprocessor

# Function to load models and data
def load_data_and_models():
    try:
        laptop_data = pd.read_csv("Laptop_log_data.csv")
        laptop_model = joblib.load('laptop_model.joblib')
        return laptop_data, laptop_model
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}")
        return None, None

laptop_data, laptop_model = load_data_and_models()

# Streamlit UI
st.title("IT Preventative Maintenance")

# Sidebar
st.sidebar.title("Options")
st.sidebar.write("Add any additional options or information here.")

# Laptop Fault Detection
with st.expander("Laptop Fault Detection"):
    if laptop_data is not None:
        st.dataframe(laptop_data.drop(columns=['Target']))

        if st.button("Scan Laptops"):
            laptop_df_x = laptop_data.drop(columns=['Laptop ID', 'Target'])
            preprocessor = LaptopDataPreprocessor()
            X_preprocessed = preprocessor.fit_transform(laptop_df_x)

            predictions = laptop_model.predict(X_preprocessed)
            result_df = laptop_data[['Laptop ID']].copy()
            result_df['Prediction'] = predictions
            result_df['Prediction_category'] = result_df['Prediction'].apply(lambda x: "Faulty" if x == 1 else "Normal")

            st.dataframe(result_df)
    else:
        st.write("No data available for analysis.")

# Additional Features
st.markdown("---")
st.write("Automatic procedure generation, and fault detectors...")

# Sidebar additional information
st.sidebar.header("About")
st.sidebar.info(
    "This app is designed to detect faults in laptops and servers using pre-trained models. "
    "Upload or enter your data for analysis."
)

# Handling real-time user inputs
with st.expander("Real-time Data Analysis"):
    # Add input widgets and process user input here
    pass

# Conclusion and additional notes
st.markdown("## Conclusion")
st.write("This tool aids in IT preventative maintenance by analyzing and predicting potential faults in laptops and servers.")

# End of Streamlit App
if __name__ == '__main__':
    st.sidebar.markdown("Developed by [Your Name/Organization]")
