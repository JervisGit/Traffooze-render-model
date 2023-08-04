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
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

mongo_uri = os.environ.get('mongoDB')
apikey = os.environ.get('apiKey')
weather_apikey = os.environ.get('openweather')

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

@app.route('/generate_predictions', methods=['GET'])
def process():

    client = pymongo.MongoClient(mongo_uri)
    db = client['TraffoozeDBS']
    collection = db['roads_metadata']

    cursor = collection.find()

    data_list = list(cursor)
    metadata_df = pd.DataFrame(data_list)

    client.close()

    current_datetime = datetime.now()

    start_date = current_datetime.replace(hour=0, minute=0, second=0, microsecond=0)

    end_date = start_date + timedelta(days=5)

    timestamps = [start_date + timedelta(minutes=i*5) for i in range(int((end_date - start_date).total_seconds() // 300))]

    prediction_data_list = []

    for index, row in metadata_df.iterrows():
        road_id = row["road_id"]
        road_length = row["length"]
        road_shape = row["shape"]
        road_start_lat = row['start_lat']
        road_start_lng = row['start_lng']
        prediction_data = pd.DataFrame({'road_id': [road_id] * len(timestamps),
                                        'length': [road_length] * len(timestamps),
                                        'shape': [road_shape] * len(timestamps),
                                        'start_lat': [road_start_lat] * len(timestamps),
                                        'start_lng': [road_start_lng] * len(timestamps),
                                        'timestamp': timestamps})
        prediction_data_list.append(prediction_data)

    combined_data = pd.concat(prediction_data_list, ignore_index=True)

    combined_data['timestamp'] = pd.to_datetime(combined_data['timestamp'])
    original_time_zone = 'Asia/Singapore'
    combined_data['timestamp'] = combined_data['timestamp'].dt.tz_localize(original_time_zone)
    combined_data['timestamp'] = combined_data['timestamp'].dt.tz_convert('GMT')

    locations = [
        {"latitude": 1.35806, "longitude": 103.940277},  # Tampines estate
        {"latitude": 1.36667, "longitude": 103.883331},  # Somapah Serangoon
        {"latitude": 1.36667, "longitude": 103.800003},  # Republic of Singapore
        {"latitude": 1.28967, "longitude": 103.850067},  # Singapore
        {"latitude": 1.41, "longitude": 103.874168},  # Seletar
        {"latitude": 1.37833, "longitude": 103.931938},  # Kampong Pasir Ris
        {"latitude": 1.42611, "longitude": 103.824173},  # Chye Kay
        {"latitude": 1.35, "longitude": 103.833328},  # Bright Hill Crescent
        {"latitude": 1.30139, "longitude": 103.797501},  # Tanglin Halt
        {"latitude": 1.44444, "longitude": 103.776672},  # Woodlands
        {"latitude": 1.35722, "longitude": 103.836388},  # Thomson Park
        {"latitude": 1.31139, "longitude": 103.797783},  # Chinese Gardens
        {"latitude": 1.35222, "longitude": 103.898064},  # Kampong Siren
        {"latitude": 1.36278, "longitude": 103.908333},  # Punggol Estate
    ]

    weather_data_dict = {}

    def get_weather_data(location, timestamp):
        latitude, longitude = location["latitude"], location["longitude"]
        api_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={latitude}&lon={longitude}&appid={weather_apikey}"
        response = requests.get(api_url)
        weather_list = response.json()["list"]
        return location, weather_list
    
    for location in locations:
        location_data, weather_list = get_weather_data(location, timestamps[0])  # Get weather data for the first timestamp
        weather_data_dict[location_data['latitude'], location_data['longitude']] = weather_list

    def calculate_distance(coord1, coord2):
        return geodesic(coord1, coord2).meters
    
    def get_closest_weather_to_timestamp(weather, timestamp):
        target_timestamp = int(datetime.timestamp(timestamp))
        closest_weather = min(weather, key=lambda x: abs(x['dt'] - target_timestamp))
        return closest_weather
    
    count = 0

    def get_weather_attributes(location, timestamp):
        weather_list = weather_data_dict[location['latitude'], location['longitude']]
        closest_weather = get_closest_weather_to_timestamp(weather_list, timestamp)
        
        global count
        
        print(count)
        count +=1
        
        return {
            'temperature': closest_weather.get('main', {}).get('temp', 0),
            'humidity': closest_weather.get('main', {}).get('humidity', 0),
            'pressure': closest_weather.get('main', {}).get('pressure', 0),
            'visibility': closest_weather.get('visibility', 0),
            'wind_speed': closest_weather.get('wind', {}).get('speed', 0),
            'wind_degree': closest_weather.get('wind', {}).get('deg', 0),
            'wind_gust': closest_weather.get('wind', {}).get('gust', 0),
            'clouds': closest_weather.get('clouds', {}).get('all', 0),
            'rain_3h': closest_weather.get('rain', {}).get('3h', 0)
        }
    
    combined_data = pd.concat([combined_data, combined_data.apply(lambda row: get_weather_attributes(min(locations, key=lambda loc: calculate_distance((row['start_lat'], row['start_lng']), (loc['latitude'], loc['longitude']))), row['timestamp']), axis=1, result_type='expand')], axis=1)

    results = combined_data.to_dict("records")

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

    geolocator = Nominatim(user_agent="myGeocoder")

    for index, row in jam.iterrows():
        date = row["Date"]
        time = row["Time"]
        message = row['Message']
        location = str(row["Latitude"]) + "," + str(row["Longitude"])
        try:
            location_info = geolocator.reverse(location, exactly_one=True)
            address = location_info.address
        except Exception as e:
            address = None

        traffic_jam = {}

        traffic_jam["date"] = date
        traffic_jam["time"] = time
        traffic_jam["message"] = message
        traffic_jam["location"] = location
        traffic_jam["address"] = address

        traffic_jams.append(traffic_jam)   

    result = collection.insert_many(traffic_jams)

    if result.inserted_ids:
        client.close()
        return "Traffic jams saved successfully."
    else:
        client.close()
        return "No traffic jams to insert."
    
@app.route('/trafficjam', methods=['GET'])
def get_trafficjam():

    client = pymongo.MongoClient(mongo_uri)

    db = client['TraffoozeDBS']
    collection = db['jvs_sample_trafficjam']

    cursor = collection.find()

    traffic_jams = []

    for document in cursor:
        document["_id"] = str(document["_id"])
        traffic_jams.append(document)

    client.close()

    return jsonify(traffic_jams)

@app.route('/save_roadclosure', methods=['GET'])
def save_roadclosure():

    client = pymongo.MongoClient(mongo_uri)

    db = client['TraffoozeDBS']
    collection = db['jvs_sample_roadclosure']

    headers = { 'AccountKey' : apikey,
             'accept' : 'application/json'}

    response = requests.get('http://datamall2.mytransport.sg/ltaodataservice/TrafficIncidents', headers=headers)
    data = response.json()["value"]
    df = pd.DataFrame(data)
    df[['Date', 'Time']] = df['Message'].str.extract(r'\((.*?)\)(.*?) ')

    df['Message'] = df['Message'].str.replace(r'\(\d+/\d+\)\d+:\d+ ', '', regex=True)

    current_year = pd.Timestamp.now().year
    df['Date'] = df['Date'] + '/' + str(current_year)

    closures = df.loc[df['Type'] == "Road Block"]

    road_closures = []

    geolocator = Nominatim(user_agent="myGeocoder")

    for index, row in closures.iterrows():
        date = row["Date"]
        time = row["Time"]
        message = row['Message']
        location = str(row["Latitude"]) + "," + str(row["Longitude"])
        try:
            location_info = geolocator.reverse(location, exactly_one=True)
            address = location_info.address
        except Exception as e:
            address = None

        road_closure = {}

        road_closure["date"] = date
        road_closure["time"] = time
        road_closure["message"] = message
        road_closure["location"] = location
        road_closure["address"] = address

        road_closures.append(road_closure)   

    result = collection.insert_many(road_closures)

    if result.inserted_ids:
        client.close()
        return "Road closures saved successfully."
    else:
        client.close()
        return "No Road closures to insert."
    
@app.route('/roadclosure', methods=['GET'])
def get_roadclosure():

    client = pymongo.MongoClient(mongo_uri)

    db = client['TraffoozeDBS']
    collection = db['jvs_sample_roadclosure']

    cursor = collection.find()

    road_closures = []

    for document in cursor:
        document["_id"] = str(document["_id"])
        road_closures.append(document)

    client.close()

    return jsonify(road_closures)

@app.route('/save_roadaccident', methods=['GET'])
def save_roadaccident():

    client = pymongo.MongoClient(mongo_uri)

    db = client['TraffoozeDBS']
    collection = db['jvs_sample_roadaccident']

    headers = { 'AccountKey' : apikey,
             'accept' : 'application/json'}

    response = requests.get('http://datamall2.mytransport.sg/ltaodataservice/TrafficIncidents', headers=headers)
    data = response.json()["value"]
    df = pd.DataFrame(data)
    df[['Date', 'Time']] = df['Message'].str.extract(r'\((.*?)\)(.*?) ')

    df['Message'] = df['Message'].str.replace(r'\(\d+/\d+\)\d+:\d+ ', '', regex=True)

    current_year = pd.Timestamp.now().year
    df['Date'] = df['Date'] + '/' + str(current_year)

    accidents = df.loc[df['Type'] == "Accident"]

    road_accidents = []

    geolocator = Nominatim(user_agent="myGeocoder")

    for index, row in accidents.iterrows():
        date = row["Date"]
        time = row["Time"]
        message = row['Message']
        location = str(row["Latitude"]) + "," + str(row["Longitude"])
        try:
            location_info = geolocator.reverse(location, exactly_one=True)
            address = location_info.address
        except Exception as e:
            address = None

        road_accident = {}

        road_accident["date"] = date
        road_accident["time"] = time
        road_accident["message"] = message
        road_accident["location"] = location
        road_accident["address"] = address

        road_accidents.append(road_accident)   

    result = collection.insert_many(road_accidents)

    if result.inserted_ids:
        client.close()
        return "Road accidents saved successfully."
    else:
        client.close()
        return "No road accidents to insert."
    
@app.route('/roadaccident', methods=['GET'])
def get_roadaccident():

    client = pymongo.MongoClient(mongo_uri)

    db = client['TraffoozeDBS']
    collection = db['jvs_sample_roadaccident']

    cursor = collection.find()

    road_accidents = []

    for document in cursor:
        document["_id"] = str(document["_id"])
        road_accidents.append(document)

    client.close()

    return jsonify(road_accidents)

if __name__ == '__main__':
    app.run(port=3000, debug=True)

#yep