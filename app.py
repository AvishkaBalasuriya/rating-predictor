from controllers.RatePredict import RatePredict
from flask import request, jsonify, Flask
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

@app.route('/api/predict', methods=['POST'])
def send_feedback():
    body = request.json
    rate_predictor = RatePredict('data/model-random-forest-final.pkl')
    results = rate_predictor.predict(body)
    data = {
        "results": results
    }
    return jsonify(data)