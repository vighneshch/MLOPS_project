# webapp.py
import streamlit as st
import pickle
from surprise import Dataset, Reader
from surprise.prediction_algorithms.co_clustering import CoClustering

def load_model():
    # Load the CoClustering model from the pickle file
    pickle_file_path = 'D:/praxis july 24/mlops/ASSIGNMENT/my assignmenrt/MLOPS_project/CoClustering_Model.pkl'
    with open(pickle_file_path, 'rb') as file:
        model = pickle.load(file)
    return model

def predict_rating(model, user_id, movie_id):
    # Make a prediction using the loaded model
    prediction = model.predict(user_id, movie_id)
    return prediction.est

def main():
    st.title("Movie Recommender System")
    
    # Sidebar for user input
    st.sidebar.header("User Input")
    user_id = st.sidebar.slider("Select User ID", 1, 610, 1)
    movie_id = st.sidebar.slider("Select Movie ID", 1, 193609, 1)

    # Load the CoClustering model
    co_clustering_model = load_model()

    # Make a prediction
    predicted_rating = predict_rating(co_clustering_model, user_id, movie_id)

    # Display the prediction
    st.write(f"Predicted Rating: {predicted_rating:.2f}")

if __name__ == "__main__":
    main()
