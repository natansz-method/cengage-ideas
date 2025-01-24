import streamlit as st
import pandas as pd

# Function to cache the uploaded file
@st.cache_data
def load_csv(file):
    """
    Load the uploaded CSV file into a DataFrame and cache it in memory.
    """
    return pd.read_csv(file)

# Main function to build the app
def main():
    st.title("CSV Data Analysis App")

    # Sidebar menu
    st.sidebar.header("Menu")
    menu = st.sidebar.radio(
        "Choose an option:",
        options=["Upload CSV", "View Data", "Statistics", "Visualizations"]
    )

    # Upload CSV file
    if menu == "Upload CSV":
        st.header("Upload Your CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file")
        
        if uploaded_file is not None:
            # Load the CSV file and cache it
            df = load_csv(uploaded_file)
            st.success("CSV file uploaded successfully!")
            st.session_state["df"] = df
            st.write("Preview of the uploaded data:")
            st.dataframe(df.head())

    # View data if already uploaded
    elif menu == "View Data":
        if "df" in st.session_state:
            st.header("View Data")
            st.write("Here is your data:")
            st.dataframe(st.session_state["df"])
        else:
            st.warning("Please upload a CSV file first.")

    # Statistics option
    elif menu == "Statistics":
        if "df" in st.session_state:
            st.header("Statistics")
            st.write("Basic statistics of the data:")
            st.write(st.session_state["df"].describe())
        else:
            st.warning("Please upload a CSV file first.")

    # Visualizations option
    elif menu == "Visualizations":
        if "df" in st.session_state:
            st.header("Visualizations")
            st.write("This is where visualizations will go.")
            st.info("Add visualization code here based on your specific needs.")
        else:
            st.warning("Please upload a CSV file first.")

# Run the app
if __name__ == "__main__":
    main()

