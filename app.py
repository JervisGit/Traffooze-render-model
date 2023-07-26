from flask import Flask, jsonify, request
import os
import pandas as pd
import joblib
#import sklearn
from flask_cors import CORS
import datetime
import requests
import json
import pymongo

app = Flask(__name__)
CORS(app)

mongo_uri = os.environ.get('mongoDB')
apikey = os.environ.get('apiKey')

my_model = joblib.load(r'rf_model.sav')

rf = joblib.load(r'rf_regressor.sav')

@app.route('/', methods=['GET'])
def hello_world():
    return "Hello Beautiful World"

@app.route('/predict', methods=['POST'])
def predict():

    features = request.get_json()

    # Extract the features for prediction
    link_id = features['LinkID']
    year = features['year']
    month = features['month']
    day = features['day']
    hour = features['hour']
    minute = features['minute']
    road_category_A = features['RoadCategory_A']
    road_category_B = features['RoadCategory_B']
    road_category_C = features['RoadCategory_C']
    road_category_D = features['RoadCategory_D']
    road_category_E = features['RoadCategory_E']
    road_category_F = features['RoadCategory_F']

    prediction = my_model.predict([[
        link_id, year, month, day, hour, minute,
        road_category_A, road_category_B, road_category_C,
        road_category_D, road_category_E, road_category_F
    ]])

    # Convert the prediction ndarray to a list
    prediction = prediction.tolist()

    response = {
        'prediction': prediction
    }

    return jsonify(response)

@app.route('/verify', methods=['GET'])
def verify():

    metadata_df = pd.read_csv('roads_metadata.csv')
    
    if not metadata_df.empty:
        return "df has values"
    else:
        return 404


@app.route('/predict/layer', methods=['GET'])
def process():

    metadata_df = pd.read_csv('roads_metadata.csv')
    metadata_df = metadata_df[["road_id", "length"]]
    timestamp = pd.to_datetime("12/07/2023 16:00")#.to_pydatetime() 

    prediction_list = []

    for index, row in metadata_df.iterrows():
        road_id = row['road_id']
        road_length = row['length']
        
        prediction_dict = {
            'road_id': road_id,
            'length': road_length,
            'Year': timestamp.year,
            'Month': timestamp.month,
            'Day': timestamp.day,
            'Hour': timestamp.hour,
            'Minute': timestamp.minute,
            'dayofweek': timestamp.dayofweek,
            'weekofyear': int(timestamp.strftime("%V")) #timestamp.isocalendar().week #isocalender()[1]
        }
        
        prediction_list.append(prediction_dict)

    prediction_df = pd.DataFrame(prediction_list)

    predictions = rf.predict(prediction_df)

    result_df = pd.DataFrame({
        'road_id': prediction_df['road_id'],
        'length': prediction_df['length'],
        'Year': prediction_df['Year'],
        'Month': prediction_df['Month'],
        'Day': prediction_df['Day'],
        'Hour': prediction_df['Hour'],
        'Minute': prediction_df['Minute'],
        'dayofweek': prediction_df['dayofweek'],
        'weekofyear': prediction_df['weekofyear'],
        'speed_prediction': predictions[:, 0],  # Assuming speed is the first target variable
        'speedUncapped_prediction': predictions[:, 1],  # Assuming speedUncapped is the second target variable
        'freeFlow_prediction': predictions[:, 2],  # Assuming freeFlow is the third target variable
        'jamFactor_prediction': predictions[:, 3]  # Assuming jamFactor is the fourth target variable
    })

    result_df["speed_prediction"] = result_df["speed_prediction"].round(1)
    result_df["speedUncapped_prediction"] = result_df["speedUncapped_prediction"].round(1)
    result_df["freeFlow_prediction"] = result_df["freeFlow_prediction"].round(1)
    result_df["jamFactor_prediction"] = result_df["jamFactor_prediction"].round(1)

    metadata_df = pd.read_csv('roads_metadata.csv')
    metadata_df = metadata_df.drop('length', axis=1)

    merged_df = pd.merge(result_df, metadata_df, on='road_id', how='left')

    merged_df.drop(["road_id", "Year", "Month", "Day", "Hour", "Minute", "dayofweek", "weekofyear"], axis=1, inplace=True)

    results = merged_df.to_dict("records")

    return jsonify(results)


@app.route('/save_trafficjam', methods=['GET'])
def save_trafficjam():

    client = pymongo.MongoClient(mongo_uri)

    db = client['TraffoozeDBS']
    collection = db['jvs_sample_trafficjam']

    headers = { 'AccountKey' : apikey,
             'accept' : 'application/json'}

    response = requests.get('http://datamall2.mytransport.sg/ltaodataservice/TrafficIncidents', headers=headers)
    data = response.json()["value"]
    df = pd.DataFrame(data)
    df[['Date', 'Time']] = df['Message'].str.extract(r'\((.*?)\)(.*?) ')

    df['Message'] = df['Message'].str.replace(r'\(\d+/\d+\)\d+:\d+ ', '', regex=True)

    current_year = pd.Timestamp.now().year
    df['Date'] = df['Date'] + '/' + str(current_year)

    jam = df.loc[df['Type'] == "Heavy Traffic"]

    traffic_jams = []

    for index, row in jam.iterrows():
        date = row["Date"]
        time = row["Time"]
        message = row['Message']
        location = row["Latitude"] + row["Longitude"]

        traffic_jam = {}

        traffic_jam["date"] = date
        traffic_jam["time"] = time
        traffic_jam["message"] = message
        traffic_jam["location"] = location

        traffic_jams.append(traffic_jam)   

    result = collection.insert_many(traffic_jams)

    if result.inserted_ids:
        client.close()
        return 200
    else:
        client.close()
        return 404
    
@app.route('/trafficjam', methods=['GET'])
def get_trafficjam():

    client = pymongo.MongoClient(mongo_uri)

    db = client['TraffoozeDBS']
    collection = db['jvs_sample_trafficjam']

    cursor = collection.find()

    traffic_jams = []

    for document in cursor:
        traffic_jams.append(document)

    client.close()

    return jsonify(traffic_jams)

if __name__ == '__main__':
    app.run(port=3000, debug=True)

#yep