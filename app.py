import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🎬 Movie Recommendation System")
st.write("Find movies similar to your favorite movie!")


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("movies.csv")

    # Remove rows where important information is missing
    df = df.dropna(subset=["title", "genre", "overview"])

    # Convert everything to string
    df["genre"] = df["genre"].astype(str)
    df["overview"] = df["overview"].astype(str)

    # Combine features
    df["features"] = (
        df["genre"] + " " +
        df["overview"]
    )

    return df


df = load_data()


# --------------------------------------------------
# CREATE TF-IDF MATRIX
# --------------------------------------------------

@st.cache_resource
def create_model(features):

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(features)

    similarity_matrix = cosine_similarity(tfidf_matrix)

    return similarity_matrix


similarity_matrix = create_model(
    df["features"]
)


# --------------------------------------------------
# MOVIE SELECTION
# --------------------------------------------------

movie_list = df["title"].tolist()

selected_movie = st.selectbox(
    "🎥 Select a movie:",
    movie_list
)


# --------------------------------------------------
# RECOMMENDATION FUNCTION
# --------------------------------------------------

def recommend_movies(movie_name, number_of_movies=5):

    # Find index of selected movie
    movie_index = df[
        df["title"] == movie_name
    ].index[0]

    # Get similarity scores
    similarity_scores = list(
        enumerate(similarity_matrix[movie_index])
    )

    # Sort movies by similarity
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    # Skip first movie because it is the movie
    # selected by the user
    for index, score in similarity_scores[1:number_of_movies + 1]:

        recommendations.append({
            "title": df.iloc[index]["title"],
            "similarity": score
        })

    return recommendations


# --------------------------------------------------
# RECOMMENDATION BUTTON
# --------------------------------------------------

if st.button("🍿 Recommend Movies"):

    recommendations = recommend_movies(
        selected_movie
    )

    st.subheader(
        f"Movies similar to **{selected_movie}**:"
    )

    for movie in recommendations:

        st.write(
            f"🎬 **{movie['title']}**"
        )

        st.progress(
            min(int(movie["similarity"] * 100), 100)
        )

        st.write(
            f"Similarity: "
            f"{movie['similarity']:.2%}"
        )

        st.divider()