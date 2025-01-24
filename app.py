import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from nltk import word_tokenize
import numpy as np

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.util import ngrams
from collections import Counter
from nltk.corpus import stopwords
import ast
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import silhouette_score, silhouette_samples

# Ensure NLTK stopwords are downloaded
nltk.download("stopwords")
nltk.download("punkt")

# Define stopwords
STOPWORDS = set(stopwords.words("english"))

# Function to cache the uploaded file
@st.cache_data
def load_csv(file):
    """
    Load the uploaded CSV file into a DataFrame and cache it in memory.
    """
    df = pd.read_csv(file)
    # Ensure 'Idea created date' is parsed as datetime
    df['Idea created date'] = pd.to_datetime(df['Idea created date'], errors='coerce')
    df['Desired delivery date'] = pd.to_datetime(df['Desired delivery date'], errors='coerce')
    # Convert all "Impact Score" columns from 0-4 to 1-5 range
    impact_columns = df.filter(regex="Impact Score").columns
    df["Embeddings"] = df["Embeddings"].apply(lambda x: np.fromstring(x, sep=","))


    for col in impact_columns:
        df[col] = df[col] + 1
    return df

# Helper function to convert WordCloud to matplotlib figure
def wordcloud_to_fig(wordcloud):
    """
    Convert a WordCloud object to a matplotlib figure for display in Streamlit.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    return fig


# Function to generate N-grams and WordCloud
def generate_wordcloud(df, column, ngram_range):
    """
    Generate a WordCloud for the given DataFrame column based on the selected N-gram range.
    """
    # Tokenize and generate N-grams
    text_data = df[column].dropna().tolist()
    tokens = [nltk.word_tokenize(text.lower()) for text in text_data]

    # Remove stopwords
    tokens = [[word for word in sentence if word not in STOPWORDS] for sentence in tokens]

    # Generate N-grams
    ngrams_list = [
        " ".join(ngram)
        for sentence in tokens
        for ngram in ngrams(sentence, ngram_range)
    ]

    # Count frequency of N-grams
    ngram_counts = Counter(ngrams_list)

    # Generate WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(ngram_counts)
    return wordcloud

# Main function to build the app
def main():

    # Sidebar menu
    st.sidebar.header("Cengage Aha! Analysis")
    menu = st.sidebar.radio(
        "Choose an option:",
        options=["Upload CSV", "Data Overview", "Most Impactful Ideas" , "Idea Clustering", "Inspect Clusters", "Search Ideas"]
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
            st.write(f"In this Aha! Data export, there are **{len(df)} unique ideas** present. The following are some distributions of these.")
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

            # New Visualization: Ideas Desired in Month by Status
            st.subheader("Ideas Desired in Month by Status")
            
            # Extract the desired month-year from the 'Desired delivery date' column
            if 'Desired delivery date' in df.columns:
                # Extract the desired month-year from the 'Desired delivery date' column
                df['desired_month_year'] = pd.to_datetime(df['Desired delivery date'], errors='coerce').dt.to_period('M').astype(str)

# Replace NaT with a placeholder string
                df['desired_month_year'].fillna("No Desired Date", inplace=True)

# Group data by desired_month_year and Idea status
                grouped_data = df.groupby(['desired_month_year', 'Idea status']).size().unstack(fill_value=0)

# Create a stacked bar chart using Plotly
                fig_desired = go.Figure()

                for status in grouped_data.columns:
                    fig_desired.add_trace(
                        go.Bar(
                            name=status,
                            x=grouped_data.index,
                            y=grouped_data[status],
                            marker_color=status_colors.get(status, "grey")
                        )
                    )

                fig_desired.update_layout(
                    barmode="stack",
                    title="Ideas Desired in Month by Status",
                    xaxis_title="Month-Year",
                    yaxis_title="Count",
                    xaxis_tickangle=45,
                    title_x=0.5,  # Center-align the title
                    legend=dict(
                        orientation="h",  # Horizontal legend
                        yanchor="top",  # Align to top of legend box
                        y=-0.2,  # Position below the chart
                        xanchor="center",  # Center-align horizontally
                        x=0.5  # Center of the chart
                    )
                )

# Plot in Streamlit
                st.plotly_chart(fig_desired, use_container_width=True)
            else:
                st.warning("'Desired delivery date' column is missing or not in datetime format.")
            # 4. Distribution of "Idea Created By"
            st.subheader("Top 10 Contributors")
            
            # Count occurrences of each contributor
            created_by_counts = df['Idea created by'].value_counts().nlargest(10)

            # Create horizontal bar chart for contributors
            fig_contributors = go.Figure(
                go.Bar(
                    x=created_by_counts.values,
                    y=created_by_counts.index,
                    orientation="h",
                    marker_color="lightblue",
                    text=[f"{count}" for count in created_by_counts.values],
                    textposition="outside"
                )
            )
            fig_contributors.update_layout(
                title="Top 10 Contributors",
                xaxis_title="Number of Ideas",
                yaxis_title="Contributors",
                yaxis=dict(autorange="reversed"),  # Largest contributor at the top
                title_x=0.5,  # Center-align the title
            )
            st.plotly_chart(fig_contributors, use_container_width=True)



            # 5. WordCloud of Most Common Words
            st.subheader("Most Common Words in Ideas")
            st.markdown(
                "Select the number of words to analyze (e.g., unigrams, bigrams, trigrams)."
            )

            # Dropdown to select N-gram range
            ngram_range = st.selectbox("Choose N-gram Range:", options=[1, 2, 3, 4, 5], index=0)

            if "Full Idea Cleaned" in df.columns:
                # Generate WordCloud
                wordcloud = generate_wordcloud(df, "Full Idea Cleaned", ngram_range)

                # Display the WordCloud
                st.pyplot(wordcloud_to_fig(wordcloud))
            else:
                st.warning("'Full Idea Cleaned' column is missing or empty.")
        else:
            st.warning("Please upload a CSV file first.")

    elif menu == "Most Impactful Ideas":
        if "df" in st.session_state:
            df = st.session_state["df"]

            st.header("Most Impactful Ideas")
            st.write("This section focuses on the distribution of impact scores across different areas.")

            # Impact Areas and Assigned Colors
            impact_areas = [
                "Total",
                "Compliance & Legal",
                "Customer Experience ",
                "Operational Efficiency",
                "Sales/Revenue ",
                "Strategic Support"
            ]
            impact_area_colors = {
                "Total": "blue",
                "Compliance & Legal": "red",
                "Customer Experience ": "green",
                "Operational Efficiency": "orange",
                "Sales/Revenue ": "purple",
                "Strategic Support": "pink"
            }

            # Calculate Mean Impact Scores and Count of Ideas with Score = 4
            impact_scores = {}
            score_4_counts = {}

            for area in impact_areas:
                column_name = f"Impact Score {area}"
                if column_name in df.columns:
                    impact_scores[area] = df[column_name].mean()  # Average score
                    score_4_counts[area] = df[df[column_name] > 4].shape[0]  # Count where score = 4

            # Create DataFrame for Plotly Bar Charts
            impact_df = pd.DataFrame({
                "Impact Area": impact_scores.keys(),
                "Average Score": impact_scores.values(),
                "Ideas Scored 4": score_4_counts.values()
            })

            # Plot 1: Average Impact Score
            st.subheader("Average Impact Scores")
            fig_avg_scores = go.Figure(
                go.Bar(
                    x=impact_df["Impact Area"],
                    y=impact_df["Average Score"],
                    name="Average Score",
                    marker_color=[impact_area_colors.get(area, "grey") for area in impact_df["Impact Area"]],
                    text=[f"{score:.2f}" for score in impact_df["Average Score"]],
                    textposition="auto"
                )
            )
            fig_avg_scores.update_layout(
                title="Average Impact Scores by Area",
                xaxis_title="Impact Area",
                yaxis_title="Average Score",
                title_x=0.5,  # Center-align title
            )
            st.plotly_chart(fig_avg_scores, use_container_width=True)

            # Plot 2: Count of Ideas Scored = 4
            st.subheader("Count of Ideas Scored 4")
            fig_count_scores = go.Figure(
                go.Bar(
                    x=impact_df["Impact Area"],
                    y=impact_df["Ideas Scored 4"],
                    name="Ideas Scored 4",
                    marker_color=[impact_area_colors.get(area, "grey") for area in impact_df["Impact Area"]],
                    text=impact_df["Ideas Scored 4"],
                    textposition="auto"
                )
            )
            fig_count_scores.update_layout(
                title="Count of Ideas Scored 4 by Area",
                xaxis_title="Impact Area",
                yaxis_title="Count of Ideas",
                title_x=0.5,  # Center-align title
            )
            st.plotly_chart(fig_count_scores, use_container_width=True)

# Visualization: Average Score vs. Idea Status
            st.subheader("Average Impact Score by Idea Status")

# Extract and clean impact scores
            impact_scores = df.filter(regex="Impact Score*").dropna()
            impact_scores.columns = impact_scores.columns.str.replace("Impact Score ", "", regex=True)

# Group by Idea Status and calculate mean
            if 'Idea status' in df.columns:
                impact_scores_by_status = impact_scores.join(df['Idea status']).groupby('Idea status').mean()

                # Add annotations to display scores in each cell
                annotations = []
                for i, row in enumerate(impact_scores_by_status.index):
                    for j, col in enumerate(impact_scores_by_status.columns):
                        annotations.append(
                            dict(
                                x=col,
                                y=row,
                                text=f"{impact_scores_by_status.iloc[i, j]:.2f}",  # Format to 2 decimal places
                                showarrow=False,
                                font=dict(color="black", size=12),  # Text styling
                            )
                        )

                # Create Plotly Heatmap
                fig_heatmap = go.Figure(
                    data=go.Heatmap(
                        z=impact_scores_by_status.values,
                        x=impact_scores_by_status.columns,
                        y=impact_scores_by_status.index,
                        colorscale="RdYlGn",
                        colorbar=dict(title="Avg Score"),
                        zmin=1,  # Adjust based on your score range (1-5 if normalized)
                        zmax=5   # Adjust based on your score range
                    )
                )

                # Add annotations to the heatmap
                fig_heatmap.update_layout(annotations=annotations)

                # Update layout for square grid
                fig_heatmap.update_layout(
                    title="Impact Scores by Idea Status and Impact Area",
                    xaxis_title="Impact Area",
                    yaxis_title="Idea Status",
                    xaxis_tickangle=45,
                    title_x=0.5,  # Center-align the title
                    autosize=False,
                    width=700,  # Set width to control aspect ratio
                    height=700  # Set height to control aspect ratio
                )

                # Show Plotly chart
                st.plotly_chart(fig_heatmap, use_container_width=False)  # Adjust container width as needed
            else:
                st.warning("'Idea status' column is missing or empty.")


            st.header("Most Impactful Ideas")
            st.write("This section focuses on the distribution of impact scores across different areas.")

            # Define the impact areas
            impact_areas = ["Total", "Compliance & Legal", "Customer Experience ", 
                            "Operational Efficiency", "Sales/Revenue ", "Strategic Support"]

            # Allow user to select the impact area
            selected_area = st.selectbox("Select Impact Area:", impact_areas)

            # Sort the DataFrame by the selected impact area's impact score
            top_ideas = df.sort_values(by=f"Impact Score {selected_area}", ascending=False)

            # Allow user to control how many ideas to display
            n = st.slider("Number of Ideas to Display:", min_value=5, max_value=50, value=10, step=5)

            # Initialize the DataFrame for the top N ideas
            temp_df = pd.DataFrame()
            temp_df["Idea Status"] = top_ideas.iloc[:n]["Idea status"].values
            temp_df["Idea Name"] = top_ideas.iloc[:n]["Idea name"].values
            temp_df["Idea Description"] = top_ideas.iloc[:n]["Idea description"].values

            for impact in impact_areas:
                temp_df[f"Impact Score {impact}"] = top_ideas.iloc[:n][f"Impact Score {impact}"].values

            temp_df[f"Impact Score {selected_area}"] = top_ideas.iloc[:n][f"Impact Score {selected_area}"].values
            temp_df["Idea created date"] = top_ideas.iloc[:n]["Idea created date"].dt.strftime('%m/%d/%y').values
            #temp_df["Desired delivery date"] = top_ideas.iloc[:n]["Desired delivery date"].dt.strftime('%m/%d/%y').values

            # Define the status_colors dictionary
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

            # Function to style Idea Status column
            def style_idea_status(value):
                color = status_colors.get(value, "transparent")
                if color != "transparent":
                    return f'background-color: {color}; color: white;'
                return ""

            # Apply the styles
            styled_df = temp_df.style.applymap(style_idea_status, subset=["Idea Status"])

            # Display the DataFrame using st.dataframe
            st.subheader(f"Top {n} Ideas for '{selected_area}'")
            st.dataframe(styled_df, use_container_width=True)

            st.divider()
            st.subheader("Wordclouds Per Area of Impact")
# Unique key added for each widget
            selected_area = st.selectbox("Select Impact Area:", impact_areas, key="impact_area_selector")

# Unique key for N-gram range selection
            ngram_range = st.selectbox("Select N-gram Range:", options=[1, 2, 3], index=0, 
                                       format_func=lambda x: f"{x}", key="ngram_range_selector")

# Unique key for number of words to display
            top_n_words = 100

            # Filter the data based on the selected impact area
            top_ideas = df[
                (df[f"Impact Score {selected_area}"] >= 4.0) &  # Score >= 4
                (df["Idea status"] != "Completed")  # Exclude completed ideas
            ]

            # Process text data for the word cloud
            processed_texts = []
            for _, row in top_ideas.iterrows():
                # Combine idea name and description
                desc = re.sub(r'http\S+', '', row["Idea description"].replace("\n", ""))
                processed_texts.append(f"{row['Idea name']} {desc}")

            # Combine all text data
            combined_text = " ".join(processed_texts)

            # Tokenize and remove stopwords
            words = word_tokenize(combined_text.lower())
            filtered_words = [word for word in words if word.isalnum() and word not in STOPWORDS]

            # Generate n-grams
            if ngram_range == 1:
                ngram_list = filtered_words
            elif ngram_range == 2:
                ngram_list = [" ".join(bigram) for bigram in zip(filtered_words, filtered_words[1:])]
            elif ngram_range == 3:
                ngram_list = [" ".join(trigram) for trigram in zip(filtered_words, filtered_words[1:], filtered_words[2:])]

            # Count frequencies of n-grams
            ngram_freq = Counter(ngram_list)

            # Generate word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white"
            ).generate_from_frequencies(dict(ngram_freq.most_common(top_n_words)))

            # Display the word cloud
            st.subheader(f"Word Cloud for {ngram_range}-grams in {selected_area}")
            st.pyplot(wordcloud_to_fig(wordcloud))

            # Display the most common n-grams
            st.subheader(f"Top {ngram_range}-grams for {selected_area}")
            for ngram, freq in ngram_freq.most_common(10):
                st.write(f"{ngram}: {freq}")


        else:
            st.warning("Please upload a CSV file first.")

# Idea Clustering Tab
    elif menu == "Idea Clustering":
        if "df" in st.session_state:
            df = st.session_state["df"]

            st.header("Idea Clustering")
            st.write("Explore how ideas are clustered based on their embeddings.")

# Ensure the embeddings are in the correct format
            if 'Embeddings' in df.columns and 'Cluster' in df.columns:
                # Convert 'Embeddings' to a 2D numpy array
                embeddings = np.vstack(df['Embeddings'].values)

                # Ensure clusters exist
                if 'Cluster' in df.columns and len(df['Cluster'].unique()) > 1:
                    # Apply LDA for dimensionality reduction
                    lda = LDA(n_components=2)
                    lda_result = lda.fit_transform(embeddings, df['Cluster'])  # Reduce to 2D

                    # Create an interactive scatter plot with Plotly
                    fig_lda = px.scatter(
                        x=lda_result[:, 0],
                        y=lda_result[:, 1],
                        color=df['Cluster'].astype(str),  # Convert clusters to strings for discrete color mapping
                        title="LDA Visualization of Ideas",
                        labels={"x": "LDA Component 1", "y": "LDA Component 2"},
                        opacity=0.7,
                        height=800,
                        hover_data={"Idea Name": df['Idea name']}  # Add "Idea name" as hover info
                    )

                    # Update marker size and text position
                    fig_lda.update_traces(marker=dict(size=8, opacity=0.8), textposition="top center")

                    # Update layout for better visualization
                    fig_lda.update_layout(
                        legend_title="Clusters",
                        title_x=0.5  # Center-align the title
                    )

                    # Display the Plotly figure in Streamlit
                    st.plotly_chart(fig_lda, use_container_width=True)
                else:
                    st.warning("The 'Cluster' column is missing or contains only one unique value. Clustering requires multiple clusters.")
            else:
                st.warning("The 'Embeddings' or 'Cluster' column is missing in the dataset. Please ensure they are included.")



        else:
            st.warning("Please upload a CSV file first.")


    # Visualizations option
    elif menu == "Inspect Clusters":
        if "df" in st.session_state:
            st.header("Inspect Clusters")
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

