"""
This file (test_service.py) contains the unit tests for Flask ML model service.
"""
import os
from pathlib import Path
import copy
import json
from utils import read_json

# Declare file path
basedir = os.path.abspath(Path(__file__).parent.parent)
json_data_path = basedir + '/data/data.json'
data = read_json(json_data_path)


def test_service_api_get(service_app):
    """
    GIVEN A FLASK app running
    WHEN the '/' page is requested (GET)
    THEN check the response is valid
    """
    response = service_app.get('/')
    assert response.headers['Content-Type'] == 'application/json'
    assert response.status_code == 200
    assert json.loads(response.data) == {'description': 'service is up', 'status': 200}


def test_service_api_post_without_data(service_app):
    """
    GIVEN A FLASK app running
    WHEN the '/' page is requested (POST)
    THEN 400 Bad Request is returned
    """
    response = service_app.post('/predict')
    assert response.headers['Content-Type'] == 'application/json'
    assert response.status_code == 400
    assert json.loads(response.data) == {'error': 'Failed to decode JSON object'}


def test_service_api_predict_missing_keys(service_app):
    """
    GIVEN A FLASK app running
    WHEN the '/predict' page is requested (POST) with missing data
    THEN error message is returned
    """
    tmp_data = copy.deepcopy(data[:1])
    tmp_data[0].pop('x1')

    response = service_app.post('/predict',
                                data=json.dumps(tmp_data),
                                content_type='application/json')
    assert response.headers['Content-Type'] == 'application/json'
    assert response.status_code == 400
    assert json.loads(response.data) == {'error': "'x1' is a required property in items -> required"}


def test_service_api_predict_wrong_data_type(service_app):
    """
    GIVEN A FLASK app running
    WHEN the '/predict' page is requested (POST) with wrong data type
    THEN error message is returned
    """
    response = service_app.post('/predict',
                                data="test",
                                content_type='application/json')
    assert response.headers['Content-Type'] == 'application/json'
    assert response.status_code == 400
    assert json.loads(response.data) == {'error': 'Failed to decode JSON object: Expecting value:'
                                                  ' line 1 column 1 (char 0)'}


def test_service_api_predict_single_raw_no_classification(service_app):
    """
    GIVEN A FLASK app running
    WHEN the '/predict' page is requested (POST) with one not classified data sample
    THEN Records does not meet classification requirements is returned
    """
    response = service_app.post('/predict',
                                data=json.dumps(data[:1]),
                                content_type='application/json')
    assert response.headers['Content-Type'] == 'application/json'
    assert response.status_code == 200
    assert json.loads(response.data) == {'message': 'Records does not meet classification requirements'}


def test_service_api_predict_single_raw_classified(service_app):
    """
    GIVEN A FLASK app running
    WHEN the '/predict' page is requested (POST) with one not classified data sample
    THEN Class classification is returned
    """
    response = service_app.post('/predict',
                                data=json.dumps(data[1:2]),
                                content_type='application/json')

    response_data = json.loads(response.data)
    assert response.headers['Content-Type'] == 'application/json'
    assert response.status_code == 200
    assert response_data['message'] == 'Records successfully classified'
    assert len(response_data['prediction'].keys()) == 102
    assert response_data['prediction']['business_outcome'] == [4]
    assert response_data['prediction']['phat'] == [0.8228085289874678]
    assert all(len(value) == 1 for value in response_data['prediction'].values())


def test_service_api_predict_multiple_raw_classified(service_app):
    """
    GIVEN A FLASK app running
    WHEN the '/predict' page is requested (POST) with one not classified data sample
    THEN multiple classifications are returned
    """
    response = service_app.post('/predict',
                                data=json.dumps(data),
                                content_type='application/json')
    response_data = json.loads(response.data)
    assert response.headers['Content-Type'] == 'application/json'
    assert response.status_code == 200
    assert response_data['message'] == 'Records successfully classified'
    assert len(response_data['prediction'].keys()) == 102
    assert response_data['prediction']['business_outcome'] == [4, 5]
    assert response_data['prediction']['phat'] == [0.8228085289874678, 0.753958838418463]
    assert all(len(value) == 2 for value in response_data['prediction'].values())
