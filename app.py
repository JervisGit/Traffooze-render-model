from flask import Flask, jsonify, request
import joblib
import sklearn

app = Flask(__name__)
my_model = joblib.load(r'rf_model.sav')

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

if __name__ == '__main__':
    app.run(port=3000, debug=True)

#yep