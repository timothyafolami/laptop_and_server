import streamlit as st
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
# import catboost
import catboost
class ServerDataPreprocessor:
    def __init__(self):
        # Define the preprocessing steps
        self.categorical_features = ['Server Status', 'Operating System', 'Server Location']
        self.categorical_transformer = OneHotEncoder(drop='first')

        self.numerical_features = ['Disk Usage (%)', 'CPU Usage (%)', 'Memory Usage (%)', 'Number of CPU Cores', 'RAM Capacity (GB)', 'Network Traffic (Mbps)', 'Disk I/O (IOPS)', 'Server Uptime (days)']
        self.numerical_transformer = StandardScaler()

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', self.categorical_transformer, self.categorical_features),
                ('num', self.numerical_transformer, self.numerical_features)
            ])

    def fit_transform(self, df):
        # Apply preprocessing to the data and fit the transformer
        X_preprocessed = self.preprocessor.fit_transform(df)
        return X_preprocessed

    def transform(self, df):
        # Apply preprocessing to the data (without fitting the transformer)
        X_preprocessed = self.preprocessor.transform(df)
        return X_preprocessed

    def save(self, file_path):
        # Save the preprocessor pipeline to a file
        joblib.dump(self.preprocessor, file_path)

    @staticmethod
    def load(file_path):
        # Load the preprocessor pipeline from a file
        preprocessor = joblib.load(file_path)
        preprocessor_instance = ServerDataPreprocessor()
        preprocessor_instance.preprocessor = preprocessor
        return preprocessor_instance

def preprocess_data(df):
    preprocessor = ServerDataPreprocessor.load('server_preprocessor_pipeline.joblib')
    return preprocessor.transform(df)

# Load the server model
server_model = joblib.load('server_model.joblib')

st.title("Predictive Monitoring")

def dataframe_with_html(df):
  return pd.DataFrame(df).style.apply(lambda row: [f'<div style="background-color: {"red" if x == 1 else "green"}; padding: 10px; border-radius: 5px; color: white;">{"Faulty" if x == 1 else "Good!"}</div>' for x in row['Prediction']], axis=1).to_html()

# ...

with st.expander("Server Model"):
    st.write("This section is for detecting faults in servers.")
    st.write("Provide the following information for multiple servers:")

    # Load your server data into a DataFrame
    server_df = pd.read_csv("Server_log_data.csv")
    server_data = server_df.copy()  # Use a copy of your server data
    st.dataframe(server_data.drop(columns=['Target']))

    # 
    server_df_x = server_df.drop(columns=['Server ID', 'Server Name', 'Target'])


    if st.button("Scan"):
        # Check if there is any data to process
        if not server_data.empty:
            X_preprocessed = preprocess_data(server_df_x)
            server_predictions = server_model.predict(X_preprocessed)

            # Create a DataFrame to display predictions
            result_df = server_df[['Server ID']].copy()
            result_df['Prediction'] = server_predictions
            result = result_df[result_df['Prediction'] == 1]
            result['Prediction_category'] = "Faulty"
        
            st.dataframe(result)


st.markdown("---")
st.write("Automatic procedure generation, and fault detectors...")

# Optionally, you can add a sidebar for additional options, if needed
st.sidebar.title("Options")
st.sidebar.write("Add any additional options or information here.")

# Run the Streamlit app
if __name__ == '__main__':
    st.sidebar.header("About")
    st.sidebar.info(
        "This app is split into 3 sections, the first is that it generates automatic procedures of doing a manual process."
        "The other aspects of this app detect faults in laptops and servers using pre-trained models. "
        "Provide the required information for each model and click the respective button to detect faults."
    )