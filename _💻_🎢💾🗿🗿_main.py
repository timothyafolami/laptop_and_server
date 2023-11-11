import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Tech Fault Detection Suite",
    page_icon="🔍",
)

# Main Page Title
st.title("Welcome to the Tech Fault Detection Suite!")

# Sidebar
st.sidebar.success("Select a page above.")

# Laptop Fault Detection Introduction
st.markdown("""
    ## Laptop Fault Detection

    In the fast-paced world of technology, maintaining your laptop in top condition is crucial. Our Laptop Fault Detection tool is here to assist you in diagnosing and identifying issues with your laptop. Leveraging advanced diagnostics, this tool helps in pinpointing problems based on your laptop's features and symptoms. It's the perfect assistant for quick and accurate fault detection, keeping your device running smoothly.

    ---
""")

# Server Fault Prediction Introduction
st.markdown("""
    ## Server Fault Prediction

    For modern businesses and IT infrastructures, server reliability is a top priority. Our Server Fault Prediction model is designed to anticipate and identify potential server issues before they become critical. By analyzing server parameters and performance data, this tool provides valuable insights into the health of your server, helping to prevent downtime and maintain seamless operations. Stay ahead of server issues with our predictive diagnostics.

    ---
""")
