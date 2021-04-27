# Deploying Flask application with gunicorn and docker containers

    .
    ├── ...
    ├── data                         
    │   ├── exercise_26_test.csv    # Test data
    │   └── exercise_26_train.csv   # Train data
    ├── models                      # Saved pickle models
    │   ├── imputer.pkl             # Imputer fitted with train data
    │   ├── logit.pkl               # Classsification model
    │   ├── mld_metadata.json       # Top ranking features 
    │   └── std_scaler.pkl          # Standard scaler 
    ├── tests                       # Unit tests and test data
    │   ├── data
    │   │   ├── data.json           # Data samples 
    │   │   └── schema.json         # JSON schema
    │   ├── functional
    │   │   └── test_service.py     # Unit test for ML service app
    │   ├── unit
    │   │    └── test_model.py      # Unit test for MODELApi library
    │   └── conftest.py             # Pytest fixtures
    ├── config.py                   # Configuration file
    ├── Dockerfile                  # Dockerfile to build docker container
    ├── model.py                    # MODELApi library to build ML model
    ├── requirements.txt            # Package requirements
    ├── run_api.sh                  # Build and run docker instance
    ├── service.py                  # Flask api app
    └── utils.py                    # Helper fucntions
