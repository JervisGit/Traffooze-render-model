from flask import Flask, jsonify, request
import pandas as pd
import joblib
import sklearn

app = Flask(__name__)
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

@app.route('/predict/layer', methods=['POST'])
def process():

    metadata_df = pd.read_csv('../roads_metadata.csv')
    metadata_df = metadata_df[["road_id", "length"]]
    timestamp = pd.to_datetime("12/07/2023 16:00")

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
            'weekofyear': timestamp.isocalendar().week
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

    metadata_df = pd.read_csv('../roads_metadata.csv')
    metadata_df = metadata_df.drop('length', axis=1)

    merged_df = pd.merge(result_df, metadata_df, on='road_id', how='left')

    merged_df.drop(["road_id", "Year", "Month", "Day", "Hour", "Minute", "dayofweek", "weekofyear"], axis=1, inplace=True)

    results = merged_df.to_dict("records")

    return jsonify(results)

if __name__ == '__main__':
    app.run(port=3000, debug=True)

#yep