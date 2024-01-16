import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms.co_clustering import CoClustering
from surprise import accuracy
import mlflow
import os
import pickle

def train_and_save_model():
     
    os.environ['GOOGLE_APPLICATION_CREDENTIALS']='D:\praxis july 24\mlops\ASSIGNMENT\my assignmenrt\MLOPS_project\master-shore-411113-41df45dfb65e.json'
    # Load the ratings data
    ratings = pd.read_csv(r'D:\praxis july 24\mlops\ASSIGNMENT\my assignmenrt\MLOPS_project\ml-latest-small\ml-latest-small\ratings.csv')

    # Load the movie metadata
    movie_meta = pd.read_csv(r'D:\praxis july 24\mlops\ASSIGNMENT\my assignmenrt\MLOPS_project\ml-latest-small\ml-latest-small\movies.csv')

    # Merge ratings with movie metadata
    ratings_movie_title = pd.merge(left=ratings, right=movie_meta, how='left', left_on='movieId', right_on='movieId')
    ratings_movie_title.drop(['timestamp', 'genres'], inplace=True, axis=1)

    # Create Surprise dataset
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings_movie_title[['userId', 'movieId', 'rating']], reader)

    # Split the data into training and test sets
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    # Create CoClustering model
    co_clustering_model = CoClustering(n_cltr_u=3, n_cltr_i=3, n_epochs=20, random_state=None, verbose=False)

    # Fit the model on the training set
    co_clustering_model.fit(trainset)

    # Specify the absolute path to save the pickle file
    pickle_file_path = r'D:\praxis july 24\mlops\ASSIGNMENT\my assignmenrt\MLOPS_project\CoClustering_Model.pkl'

    # Save the CoClustering model as a pickle file
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(co_clustering_model, file)

    # Make predictions on the test set
    predictions = co_clustering_model.test(testset)

    # Evaluate the model's performance
    rmse = accuracy.rmse(predictions)

    # Log metrics and model with MLflow
    mlflow.set_tracking_uri('http://34.100.213.14:5000/')
    exp_name = 'Movie_recommender_system'

    # Creating experiment in MLflow
    if mlflow.get_experiment_by_name(exp_name) is None:
        mlflow.create_experiment(exp_name)
    else:
        mlflow.set_experiment(exp_name)

    # Creating run for CoClustering
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param('n_cltr_u', 3)
        mlflow.log_param('n_cltr_i', 3)
        mlflow.log_param('n_epochs', 20)

        # Log metrics
        mlflow.log_metric('rmse', rmse)

        # Log the CoClustering model
        mlflow.sklearn.log_model(co_clustering_model, 'CoClustering_Model')

if __name__ == "__main__":
    train_and_save_model()