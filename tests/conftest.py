import os
import pytest
from model import MODELApi
from service import app

# Model directory structure
BASEDIR = os.path.abspath(os.path.dirname('__file__'))
DATA_DIR = 'data/'
TRAIN_DATA_PATH = os.path.join(BASEDIR, DATA_DIR, 'exercise_26_train.csv')
TEST_DATA_PATH = os.path.join(BASEDIR, DATA_DIR, 'exercise_26_test.csv')


@pytest.fixture(scope='module')
def new_model_api():
    model_api = MODELApi(TRAIN_DATA_PATH, TEST_DATA_PATH)
    return model_api


@pytest.fixture(scope='module')
def class_model_api():
    return MODELApi


@pytest.fixture(scope='module')
def service_app():
    # Create a test client using the Flask application
    with app.test_client() as testing_client:
        # Establish an application context
        with app.app_context():
            yield testing_client
