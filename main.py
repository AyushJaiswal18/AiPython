import pandas as pd
from surprise import SVD,accuracy
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask,jsonify,request

def load_data():
    df_customers = pd.read_csv('customer_data.csv')
    df_scooters = pd.read_csv('scooter_data.csv')
    df_interactions = pd.read_csv('interaction_data.csv')
    return df_customers, df_scooters, df_interactions


def prepare_cf_data(df_interactions):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df_interactions[['Customer_ID', 'Scooter_ID', 'Rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.25)
    return data, trainset, testset


def train_svd_model(trainset):
    model = SVD()
    model.fit(trainset)
    return model

def evaluate_model(model, testset):
    predictions = model.test(testset)
    return accuracy.rmse(predictions)

def normalize_scooter_features(df_scooters):
    scaler = MinMaxScaler()
    df_scooters[['Battery_Capacity', 'Range', 'Price']] = scaler.fit_transform(df_scooters[['Battery_Capacity', 'Range', 'Price']])
    return df_scooters, scaler

def calculate_content_based_scores(df_scooters, user_preference, scaler):
    user_preference_normalized = scaler.transform([user_preference])
    scooter_features = df_scooters[['Battery_Capacity', 'Range', 'Price']].values
    similarity_scores = cosine_similarity(user_preference_normalized, scooter_features).flatten()
    df_scooters['Similarity_Score'] = similarity_scores
    return df_scooters

def predict_ratings(model, df_scooters, user_id):
    df_scooters['Predicted_Rating'] = 0  # Initialize
    for index, row in df_scooters.iterrows():
        scooter_id = row['Scooter_ID']
        predicted_rating = model.predict(uid=user_id, iid=scooter_id).est
        df_scooters.at[index, 'Predicted_Rating'] = predicted_rating
    return df_scooters

def calculate_hybrid_scores(df_scooters, weight_cf=0.3, weight_cb=0.7):
    df_scooters['Hybrid_Score'] = (weight_cf * df_scooters['Predicted_Rating']) + (weight_cb * df_scooters['Similarity_Score'])
    df_scooters.sort_values(by='Hybrid_Score', ascending=False, inplace=True)
    return df_scooters

def get_top_recommendations(df_scooters, top_n=5):
    return df_scooters[['Scooter_ID',"Brand","Model","Battery_Capacity","Range","Price", 'Hybrid_Score']].head(top_n)

def recommend(battery=0.2, range=0.9, price=0.9):
    battery=battery/100
    range=range/100 
    price=price/100
    user_preference_vector = np.array([battery, range, price])
    
    df_customers, df_scooters, df_interactions = load_data()
    df_customers1, df_scooters1, df_interactions1 = load_data()
    data, trainset, testset = prepare_cf_data(df_interactions)
    model = train_svd_model(trainset)
    evaluate_model(model, testset)
    
    df_scooters, scaler = normalize_scooter_features(df_scooters)
    df_scooters = calculate_content_based_scores(df_scooters, user_preference_vector, scaler)
    df_scooters = predict_ratings(model, df_scooters, user_id=1)  # Adjust user_id as needed
    df_scooters = calculate_hybrid_scores(df_scooters)

    top_recommendations = get_top_recommendations(df_scooters)
    temp=[]
    for val in top_recommendations.iterrows():
        temp.append(getRowData(val[1]["Scooter_ID"]))
        
    return temp

def getRowData(id):
    df_customers1, df_scooters1, df_interactions1 = load_data()
    for row in df_scooters1.iterrows():
        if row[1]["Scooter_ID"] == id:
            return row[1].to_json()


app = Flask(__name__)

@app.route('/', methods=['POST'])
def handle_post_request():
    # Get JSON data from the request
    data = request.get_json()

    # Extract parameters from JSON data
    param1 = data.get('battery')
    param2 = data.get('range')
    param3 = data.get('price')
    print(param1,param2,param3)

    # Call the function with the extracted parameters
    result = recommend(battery=param1, range=param2, price=param3)  
    return jsonify({'result': result})