import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Function to cache the uploaded file
@st.cache_data
def load_csv(file):
    """
    Load the uploaded CSV file into a DataFrame and cache it in memory.
    """
    df = pd.read_csv(file)
    # Ensure 'Idea created date' is parsed as datetime
    df['Idea created date'] = pd.to_datetime(df['Idea created date'], errors='coerce')
    return df

# Main function to build the app
def main():

    # Sidebar menu
    st.sidebar.header("Menu")
    menu = st.sidebar.radio(
        "Choose an option:",
        options=["Upload CSV", "Data Overview", "Idea Clustering", "Inspect Clusters", "Search Ideas"]
    )

    # Upload CSV file
    if menu == "Upload CSV":
        st.title("Cengage Work (Ed2go)")
        st.markdown("#### Aha! Ideas Analysis")
        st.write("The following is an interactive report of Aha! Ed2go Ideas. Upload the provided csv file and look through the sidebar to see more!")

        st.divider()
        st.header("Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file")
        
        if uploaded_file is not None:
            # Load the CSV file and cache it
            df = load_csv(uploaded_file)
            st.success("CSV file uploaded successfully!")
            st.session_state["df"] = df

    # View data if already uploaded
    elif menu == "Data Overview":
        if "df" in st.session_state:
            df = st.session_state["df"]

            st.header("Data Overview")
            st.write(f"In this Aha! Data export, there are {len(df)} unique ideas present. The following are some distributions of these.")
            st.divider()

            # 1. Idea Status Breakdown
            st.subheader("Idea Status Breakdown")
            
            # Count the statuses
            status_counts = df["Idea status"].value_counts()

            # Calculate percentages
            total_count = status_counts.sum()
            percentages = (status_counts / total_count) * 100

            # Define colors mapped to specific statuses
            status_colors = {
                'Planned': 'royalblue',
                'Completed': 'green',
                'Needs Research': 'yellow',
                'New - Needs Review': 'orange',
                'Future Consideration': 'purple',
                'Will not Implement': 'red',
                'Planning to Implement': 'lightgreen',
                'Already Exists': 'grey'
            }

            # Sort statuses by count in descending order
            status_counts = status_counts.sort_values(ascending=False)  # For biggest bar on top

            # Create horizontal bar chart using Plotly
            fig_status = go.Figure(
                go.Bar(
                    x=status_counts.values,
                    y=status_counts.index,
                    orientation="h",
                    marker=dict(
                        color=[status_colors.get(status, "grey") for status in status_counts.index]
                    ),
                    text=[f"{count} ({percentage:.1f}%)" for count, percentage in zip(status_counts, percentages)],
                    textposition="outside"
                )
            )
            fig_status.update_layout(
                title="Idea Status Breakdown",
                xaxis_title="Count",
                yaxis_title="Idea Status",
                yaxis=dict(autorange="reversed"),  # Invert y-axis for largest bar at the top
                title_x=0.5,  # Center the title
            )
            st.plotly_chart(fig_status, use_container_width=True)

            # 2. Ideas Created Over Time
            st.subheader("Ideas Created Over Time")
            
            # Extract the month-year from the 'Idea created date' column
            if 'Idea created date' in df.columns:
                df['month_year'] = df['Idea created date'].dt.to_period('M').astype(str)

                # Count occurrences for each month-year
                month_year_counts = df['month_year'].value_counts().sort_index()

                # Plot the interactive bar chart using Plotly
                fig_time = px.bar(
                    x=month_year_counts.index,
                    y=month_year_counts.values,
                    labels={"x": "Month-Year", "y": "Count"},
                    title="Ideas/Painpoints Per Month-Year",
                )
                fig_time.update_layout(
                    xaxis_title="Month-Year",
                    yaxis_title="Count",
                    xaxis_tickangle=45,
                    title_x=0.5,  # Center-align the title
                )
                st.plotly_chart(fig_time, use_container_width=True)

                # 3. Stacked Bar Chart for Month-Year by Status
                st.subheader("Ideas Per Month-Year by Status")
                
                # Group data by month_year and Idea status
                grouped_data = df.groupby(['month_year', 'Idea status']).size().unstack(fill_value=0)

                # Create a stacked bar chart using Plotly
                fig_stacked = go.Figure()

                for status in grouped_data.columns:
                    fig_stacked.add_trace(
                        go.Bar(
                            name=status,
                            x=grouped_data.index,
                            y=grouped_data[status],
                            marker_color=status_colors.get(status, "grey")
                        )
                    )
                fig_stacked.update_layout(
                    barmode="stack",
                    title="Ideas/Painpoints Per Month-Year by Status",
                    xaxis_title="Month-Year",
                    yaxis_title="Count",
                    xaxis_tickangle=45,
                    title_x=0.5,  # Center the title
                    legend=dict(
                        orientation="h",  # Horizontal legend
                        yanchor="top",  # Align to top of legend box
                        y=-0.2,  # Position below the chart
                        xanchor="center",  # Center-align horizontally
                        x=0.5  # Center of the chart
                    )
                )

                st.plotly_chart(fig_stacked, use_container_width=True)
            else:
                st.warning("'Idea created date' column is missing or not in datetime format.")
        else:
            st.warning("Please upload a CSV file first.")

    # Statistics option
    elif menu == "Idea Clustering":
        if "df" in st.session_state:
            st.header("Statistics")
            st.write("Basic statistics of the data:")
            st.write(st.session_state["df"].describe())
        else:
            st.warning("Please upload a CSV file first.")

    # Visualizations option
    elif menu == "Inspect Clusters":
        if "df" in st.session_state:
            st.header("Visualizations")
            st.write("This is where visualizations will go.")
            st.info("Add visualization code here based on your specific needs.")
        else:
            st.warning("Please upload a CSV file first.")

    # Visualizations option
    elif menu == "Search Ideas":
        if "df" in st.session_state:
            st.header("Visualizations")
            st.write("This is where visualizations will go.")
            st.info("Add visualization code here based on your specific needs.")
        else:
            st.warning("Please upload a CSV file first.")

# Run the app
if __name__ == "__main__":
    main()

