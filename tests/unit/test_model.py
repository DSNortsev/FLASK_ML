"""
This file (test_model.py) contains the unit tests for the model.py file.
"""
import os
from pathlib import Path
import pandas as pd
from utils import read_json

# Declare file path
basedir = os.path.abspath(Path(__file__).parent.parent)
json_data_path = basedir + '/data/data.json'


def test_model_api_class_attributes(new_model_api):
    """
    GIVEN A MODELApi library
    WHEN a new insatnce is initiated
    THEN all class parameters are correctly defined
    """
    # model_api = MODELApi(TRAIN_DATA_PATH, TEST_DATA_PATH)
    assert new_model_api.TOTAL_FEATURES == 25
    assert new_model_api.CATEGOROCAL_FEATURES == ['x5', 'x31', 'x81', 'x82']
    assert not new_model_api.DUMMY_CATEGORIES


def test_model_api_instance_parameters(new_model_api):
    """
    GIVEN A MODELApi library
    WHEN a new instance is initiated
    THEN all instance parameters are correctly defined
    """
    assert isinstance(new_model_api.train_data, pd.DataFrame)
    assert isinstance(new_model_api.test_data, pd.DataFrame)
    assert new_model_api.train_data.shape == (40000, 101)
    assert new_model_api.test_data.shape == (10000, 100)


def test_model_api_remove_special_characters(new_model_api):
    """
    GIVEN A MODELApi library
    WHEN _remove_special_characters method is invoked with Pandas DataFrame
    THEN special characters are removed from x12 and x63 columns
    """
    df = new_model_api.train_data.copy(deep=True)
    new_model_api._remove_special_characters(df)
    assert df['x12'].dtype.name == 'float64'
    assert df['x63'].dtype.name == 'float64'


def test_model_api_replace_nan(new_model_api):
    """
    GIVEN A MODELApi library
    WHEN _replace_nan method is invoked with Pandas DataFrame
    THEN all missing values are imputed with mean value
    """
    df = new_model_api.train_data.copy(deep=True)
    new_model_api._remove_special_characters(df)
    numerical_df = df[df.columns.difference(new_model_api.CATEGOROCAL_FEATURES)]
    df_imputed = new_model_api._replace_nan(numerical_df)
    assert df_imputed.isnull().values.sum() == 0


def test_model_api_normalize_numerical_data(new_model_api):
    """
    GIVEN A MODELApi library
    WHEN normalize_numerical_data ls invoked with Pandas DataFrame
    THEN all numerical values are normalized in Pandas DataFrame
    """
    df = new_model_api.test_data.copy(deep=True)
    new_model_api._remove_special_characters(df)
    numerical_df = df[df.columns.difference(new_model_api.CATEGOROCAL_FEATURES)]
    df_imputed = new_model_api._replace_nan(numerical_df)
    numerical_df = new_model_api._normalize_numerical_data(df_imputed, train=False)
    assert numerical_df['x1'].max() < 5


def test_model_api_preprocess_data(new_model_api):
    """
    GIVEN A MODELApi library
    WHEN preprocess_data ls invoked with test paramter set to True
    THEN test data is preprocessed and normalized
    """
    df_train = new_model_api.preprocess_data(train=True)
    assert df_train.shape == (40000, 122)
    assert df_train.isnull().values.sum() == 0


def test_model_api_preprocess_df(class_model_api):
    """
    GIVEN A MODELApi library
    WHEN preprocess_df ls invoked with Pandas DataFrame
    THEN test data is preprocessed and normalized
    """
    json_data = read_json(json_data_path)
    df = pd.json_normalize(json_data)
    df_processed = class_model_api.preprocess_df(df)
    assert df_processed.shape == (10, 121)
    assert df_processed.isnull().values.sum() == 0


def test_model_api_predict(class_model_api):
    """
    GIVEN A MODELApi library
    WHEN predict method ls invoked with Pandas DataFrame
    THEN Pandas DataFrame with valid classification is returned
    """
    json_data = read_json(json_data_path)
    df = pd.json_normalize(json_data)
    df_processed = class_model_api.preprocess_df(df)
    model = class_model_api.import_model('logit.pkl')
    prediction = class_model_api.predict(model, df_processed)

    assert isinstance(prediction, pd.DataFrame)
    assert prediction.shape == (10, 2)
    assert prediction.columns.tolist() == ['phat', 'business_outcome']
    assert prediction['phat'].isnull().values.sum() == 0
    assert prediction['business_outcome'].isnull().values.sum() == 8
