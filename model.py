"""ModelApi to build ML models"""
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from utils import read_json, save_json
import statsmodels.api as sm
import pickle


# Model directory structure
BASEDIR = os.path.abspath(os.path.dirname('__file__'))
DATA_DIR = 'data/'
TRAIN_DATA_PATH = os.path.join(BASEDIR, DATA_DIR, 'exercise_26_train.csv')
TEST_DATA_PATH = os.path.join(BASEDIR, DATA_DIR, 'exercise_26_test.csv')
MODEL_DIR = os.path.join(BASEDIR, 'models/')
TEST_DIR = os.path.join(BASEDIR, 'tests/')
TEST_DATA = os.path.join(TEST_DIR, 'data/')
METADATA_PATH = os.path.join(BASEDIR, 'models/', 'mld_metadata.json')
IMPUTER = 'imputer.pkl'
SCALER = 'std_scaler.pkl'
JSON_TEST = os.path.join(TEST_DATA, 'data.json')


class MODELApi:
    TOTAL_FEATURES = 25
    CATEGOROCAL_FEATURES = ['x5', 'x31', 'x81', 'x82']
    DUMMY_CATEGORIES = {}

    def __init__(self, train_data: str, test_data: str):
        self.train_data = pd.read_csv(train_data)
        self.test_data = pd.read_csv(test_data)

    def _remove_special_characters(self, df: pd.DataFrame, inplace=False) -> None:
        """Removes special characters from categorical values"""
        if not inplace:
            # Create deep copy when inplace set to False
            data = df.copy(deep=True)
        else:
            data = df

        col_fix = {'x12': [('$', ''), (',', ''), (')', ''), ('(', '-')],
                   'x63': [('%', '')]}
        for key, val in col_fix.items():
            for s_char in val:
                data[key] = data[key].str.replace(*s_char)
            df[key] = df[key].astype(float)

    def _replace_nan(self, df: pd.DataFrame, train=True) -> pd.DataFrame:
        """Replace NaN values with mean value"""
        if train:
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputer.fit(df)
        else:
            imputer = self.import_model(IMPUTER)
        df_imputed = pd.DataFrame(imputer.transform(df),
                                  columns=df.columns)

        if train:
            self.save_model(imputer, IMPUTER)
        return df_imputed

    def _normalize_numerical_data(self, df: pd.DataFrame, train=True) -> pd.DataFrame:
        """Scale numerical data with StandardScaler"""
        if train:
            std_scaler = StandardScaler()
            std_scaler.fit(df)
        else:
            std_scaler = self.import_model(SCALER)
        df_scaled = pd.DataFrame(std_scaler.transform(df), columns=df.columns)

        if train:
            self.save_model(std_scaler, SCALER)
        return df_scaled

    def preprocess_data(self, train=True) -> pd.DataFrame:
        """Pre-process raw data"""
        # Create Deep copy of Pandas DataFrame based ob data type
        if train:
            df = self.train_data.copy(deep=True)
            drop_features = self.CATEGOROCAL_FEATURES + ['y']
            # Replace Nan the most frequent value in the column
            df['y'].fillna(df['y'].mode().iloc[0], inplace=True)
        else:
            df = self.test_data.copy(deep=True)[:5]
            drop_features = self.CATEGOROCAL_FEATURES

        # 1. Removing special characters
        self._remove_special_characters(df, inplace=True)

        # 2. Drop categorical features
        numerical_df = df[df.columns.difference(drop_features)]

        # 3. Replace NaN values with mean
        numerical_df = self._replace_nan(numerical_df, train=train)
        # Replace Nan the most frequent value in the column

        # 4. Normalize numerical with
        numerical_df = self._normalize_numerical_data(numerical_df, train=train)
        df_processed = numerical_df

        # 3 Create dummies
        categorical_df = pd.DataFrame()
        for col in self.CATEGOROCAL_FEATURES:
            if train:
                dump = pd.get_dummies(df[col], drop_first=True, prefix=col,
                                      prefix_sep='_', dummy_na=True)

                # Create dummy categories for train data with missing values
                self.DUMMY_CATEGORIES[col] = [col.split('_')[1] for col in dump.columns[:-1]]
            else:
                categorical_df[col] = df[col].astype(CategoricalDtype(self.DUMMY_CATEGORIES[col]))
                dump = pd.get_dummies(categorical_df[col], prefix=col,
                                      prefix_sep='_', dummy_na=True)

            df_processed = pd.concat([df_processed, dump], axis=1, sort=False)

        if train:
            # Add y feature back to DataFrame when training the model
            df_processed = pd.concat([df_processed, df['y']], axis=1, sort=False)
            # Save model metadata for preprocessing
            save_json(self.DUMMY_CATEGORIES, METADATA_PATH)

        return df_processed

    @classmethod
    def preprocess_df(cls, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy(deep=True)
        """Pre-process from Pandas DataFrame"""
        dummy_categories = read_json(METADATA_PATH)
        drop_features = cls.CATEGOROCAL_FEATURES

        # 1. Removing special characters
        cls._remove_special_characters(cls, df, inplace=True)

        # 2. Drop categorical features
        numerical_df = df[df.columns.difference(drop_features)]

        # 3. Replace NaN values with mean
        numerical_df = cls._replace_nan(cls, numerical_df, train=False)
        # Replace Nan the most frequent value in the column

        # 4. Normalize numerical with
        numerical_df = cls._normalize_numerical_data(cls, numerical_df, train=False)
        df_processed = numerical_df

        # 3 Create dummies
        categorical_df = pd.DataFrame()
        for col in cls.CATEGOROCAL_FEATURES:
            categorical_df[col] = df[col].astype(CategoricalDtype(dummy_categories[col]))
            dump = pd.get_dummies(categorical_df[col], prefix=col,
                                  prefix_sep='_', dummy_na=True)

            df_processed = pd.concat([df_processed, dump], axis=1, sort=False)
        return df_processed

    def find_important_features(self, df: pd.DataFrame) -> list:
        exploratory_lr = LogisticRegression(penalty='l1', fit_intercept=False, solver='liblinear')
        exploratory_lr.fit(df.drop(columns=['y']), df['y'])
        exploratory_results = pd.DataFrame(df.drop(columns=['y']).columns).rename(columns={0: 'name'})
        exploratory_results['coefs'] = exploratory_lr.coef_[0]
        exploratory_results['coefs_squared'] = exploratory_results['coefs'] ** 2
        var_reduced = exploratory_results.nlargest(self.TOTAL_FEATURES, 'coefs_squared')
        TOP_RANKING_FEATURES = var_reduced['name'].to_list()
        return TOP_RANKING_FEATURES

    def build_model(self, df: pd.DataFrame) -> sm.Logit:
        """Build Logit model"""
        # Find most ranking feature
        TOP_RANKING_FEATURES = self.find_important_features(df)
        # Create Logistic Regression model
        logit = sm.Logit(df['y'], df[TOP_RANKING_FEATURES])
        # Fit the model
        model = logit.fit(disp=False)
        return model

    def save_model(self, model, name) -> None:
        """Save ML model, Imputer, Scaler"""
        with open(MODEL_DIR + name, 'wb+') as f:
            pickle.dump(model, f)

    @staticmethod
    def import_model(name):
        """Import model, Imputer, Scaler"""
        with open(MODEL_DIR + name, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def validate_model(estimator, df: pd.DataFrame):
        """Validate model"""
        # Get trained features
        TOP_RANKING_FEATURES = estimator.params.index
        prediction = pd.DataFrame(estimator.predict(df[TOP_RANKING_FEATURES])) \
            .rename(columns={0: 'probs'})
        prediction['y'] = df['y']
        # print('The C-Statistics is ', roc_auc_score(prediction['y'], prediction['probs']))
        prediction['prob_bin'] = pd.qcut(prediction['probs'], q=20)
        print(prediction)

    @staticmethod
    def predict(model, df):
        """Classifier. Return probability value and classified category"""
        top_bins = 5
        TOP_RANKING_FEATURES = model.params.index
        prediction = model.predict(df[TOP_RANKING_FEATURES])
        # Create bin with probability of 0.75 to 1
        bins = np.linspace(0.75, 1, num=top_bins+1)
        # Create classification labels
        labels = [*range(5, 0, -1)]
        predicted_category = pd.cut(prediction, bins=bins, labels=labels)
        return pd.DataFrame({'phat': prediction,
                             'business_outcome': predicted_category})


if __name__ == '__main__':
    # Initialize Model API
    model_api = MODELApi(TRAIN_DATA_PATH, TEST_DATA_PATH)

    # Normalize Train and Test data
    df_train = model_api.preprocess_data(train=True)
    df_test = model_api.preprocess_data(train=False)

    # Create Logistic Regression model
    model = model_api.build_model(df_train)
    # Save Logit model
    model_api.save_model(model, 'logit.pkl')

    # # Create Test Data
    raw_data = model_api.test_data[:10]
    raw_data.to_json(JSON_TEST, orient='records', indent=4)
