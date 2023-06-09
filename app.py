from flask import Flask
import pickle

app = Flask(__name__)
#diabetes_model = pickle.load(open('diabetes_model.sav','rb'))

@app.route('/', methods=['GET'])
def hello_world():
    return "Hello Beautiful World"

if __name__ == '__main__':
    app.run(port=3000, debug=True)

#yep