import pandas as pd
import pickle as pkl
import streamlit as st
import re
import requests
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

@st.cache_data
def load_movies():
    df = pd.read_csv("movies.csv")
    df["title"] = df["title"].str.lower()    return df

df = load_movies()

@st.cache_resource
def load_models():
    with open("knn_model.pkl", "rb") as file:
        knn_model = pkl.load(file)
    with open("movie_embeddings.pkl", "rb") as file:
        movie_embeddings = pkl.load(file)
    return knn_model, movie_embeddings

knn_model, movie_embeddings = load_models()

TMDB_API_KEY = "80d2c2c357263ec556bdeb1f2bf77a18"

@st.cache_data
def fetch_movie_details(movie_title):
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
    response = requests.get(search_url)

    if response.status_code == 200:
        data = response.json()
        if data["results"]:
            movie_id = data["results"][0]["id"]
            poster_path = data["results"][0].get("poster_path")
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
            return movie_id, poster_url
    return None, None

def get_recommendations(movie_title):
    movie_title = movie_title.lower()

    if movie_title not in df["title"].values:
        return [], []
    
    idx = df[df["title"] == movie_title].index[0]
    distances, indices = knn_model.kneighbors([movie_embeddings[idx]], n_neighbors=10)

    recommended_movies = []
    posters = []

    for i in indices.flatten()[1:]:
        recommended_movie = df.iloc[i]["title"].title()
        _, poster_url = fetch_movie_details(recommended_movie)
        recommended_movies.append(recommended_movie)
        posters.append(poster_url)

    return recommended_movies, posters

def resize_image(image_url, width=180, height=270):
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        return img.resize((width, height))
    except:
        return None

st.markdown("""
    <style>
        .stApp { background-color: black !important; }
        .main-title { font-size: 40px; font-weight: bold; color: white; text-align: center; }
        .input-container { text-align: center; margin-top: 20px; }
        .movie-title { font-size: 18px; font-weight: bold; color: white; text-align: center; }
        .back-button-container { display: flex; justify-content: center; margin-top: 30px; }
        .back-button { background-color: red; color: white; font-size: 16px; padding: 12px 20px; border-radius: 8px; }
        .back-button:hover { background-color: darkred; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🎬 Movie Recommendation System 🎬</div>", unsafe_allow_html=True)

if "recommend_clicked" not in st.session_state:
    st.session_state.recommend_clicked = False

def recommend_movies():
    st.session_state.recommend_clicked = True

st.markdown("<div class='input-container'>", unsafe_allow_html=True)
selected_movie_name = st.text_input(
    " ", placeholder="Enter a movie name...", key="movie_input", on_change=recommend_movies
)
st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.recommend_clicked and selected_movie_name:
    if not re.match(r'^[A-Za-z0-9 ]*$', selected_movie_name):
        st.error("❌ Only letters (A-Z, a-z) and numbers (0-9) are allowed!")
    else:
        spinner = st.empty()
        spinner.markdown("<div class='main-title'>🔄 Fetching Recommendations...</div>", unsafe_allow_html=True)

        recommendations, posters = get_recommendations(selected_movie_name)

        spinner.empty()

        if not recommendations:
            st.error("❌ Movie not found! Please try another.")
        else:
            st.subheader("Recommended Movies 🎥")

            NO_IMAGE_URL = "https://github.com/movie-recommendation-team/Movie-Recommendation-System/blob/main/No%20img.jpg?raw=true"

            poster_width = 180
            poster_height = 270

            cols = st.columns(4)

            for i, (movie, poster) in enumerate(zip(recommendations, posters)):
                col = cols[i % 4]
                with col:
                    if poster:
                        resized_poster = resize_image(poster, width=poster_width, height=poster_height)
                        st.image(resized_poster, use_container_width=True)
                    else:
                        resized_poster = resize_image(NO_IMAGE_URL, width=poster_width, height=poster_height)
                        st.image(resized_poster, use_container_width=True)
                    st.write(f"<div class='movie-title'>{movie}</div>", unsafe_allow_html=True)

            with st.container():
                st.markdown("<div class='back-button-container'>", unsafe_allow_html=True)
                if st.button("Back to Home"):
                    st.session_state.recommend_clicked = False
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
