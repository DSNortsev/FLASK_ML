
"""
Flask application to serve Machine Learning models
"""

import os
import pandas as pd
from flask import Flask, jsonify, request, make_response
from logging.handlers import RotatingFileHandler
import logging
from time import time
from model import MODELApi
from utils import read_json
from flask_expects_json import expects_json
from jsonschema import ValidationError

# Read env variables
DEBUG = os.environ.get('DEBUG', True)
HOST = os.environ.get('HOST', 'localhost')
PORT = os.environ.get('PORT', '1313')
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'local')

# PORT = os.environ.get('PORT', '1313')
FILE_NAME = __file__.rsplit(".", 1)[0]
SERVICE_START_TIMESTAMP = time()
# Create Flask Application
app = Flask(__name__)

# Declare file path
basedir = os.path.abspath(os.path.dirname(__file__))
models_dir = os.path.join(basedir, 'models')
json_schema_dir = basedir + '/test/schema.json'
schema = read_json(json_schema_dir)
# Load Logistic Regression model
model = MODELApi.import_model('logit.pkl')


app.logger.setLevel(logging.DEBUG if DEBUG else logging.ERROR)
handler = RotatingFileHandler(f'{FILE_NAME}.log', maxBytes=10000, backupCount=1)
app.logger.addHandler(handler)
app.logger.info(f'ENVIRONMENT: {ENVIRONMENT}')
app.logger.info(f'HOST: {HOST}')
app.logger.info(f'PORT: {PORT}')
app.logger.info(f'DEBUG: {DEBUG}')
app.logger.info('Loading model...')


@app.errorhandler(400)
def bad_request(error):
    """Error handling during json schema validation"""
    if isinstance(error.description, ValidationError):
        original_error = error.description
        return make_response(jsonify({'error': f'{original_error.message} in '
                                               + ' -> '.join(original_error.schema_path)}), 400)

    return make_response(jsonify({'error': error.description}), 400)


@app.route('/', methods=['GET'])
def server_is_up():
    """API request to check the status of the service"""
    return jsonify({"status": 200,
                    'description': 'service is up'})


@app.route('/predict', methods=['POST'])
@expects_json(schema)
def predict():
    """Predict and classify the data"""
    # Read JSON data
    json_data = request.get_json()
    # Convert JSON data to Pandas DataFrame
    df = pd.json_normalize(json_data)
    #  Normalize data
    df_processed = MODELApi.preprocess_df(df)
    # Predict data
    prediction = MODELApi.predict(model, df_processed)
    # Merge DataFrames
    pred_result = pd.merge(prediction, df, left_index=True, right_index=True) \
        .dropna(subset=['business_outcome'])

    # Check if prediction DataFrame is not empty
    if pred_result.empty:
        return jsonify({"message": 'Records does not meet classification requirements'})

    # Construct prediction response
    response_json = {"message": "Records successfully classified",
                     "prediction": dict(sorted(pred_result.to_dict('list').items()))}

    return jsonify(response_json)

@app.before_request
def before_request():
    app.logger.debug('\nREQUEST:')
    app.logger.debug('Headers: %s', request.headers)
    app.logger.debug('Body: %s', request.get_data())


# @app.after_request
# def after_request(response):
#     app.logger.debug('\nRESPONDS:')
#     # app.logger.debug('Headers: %s', response.headers)
#     # app.logger.debug('STATUS: %s', response.status)
#     # app.logger.debug('BODY: %s', response.get_data())


if __name__ == '__main__':
    app.run(
        debug=DEBUG,
        host=HOST,
        port=PORT)
