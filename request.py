import os
import requests
from utils import read_json


# Declare file path
basedir = os.path.abspath(os.path.dirname(__file__))
json_data = basedir + '/test/data.json'


data = read_json(json_data)


url = 'http://localhost:5000/predict'
r = requests.post(url, headers={"content-type": "application/json"}, json=data[:10])
print(r.json())
print(r.status_code)

# print(r.content)
# print(r.content)
# print(r.status_code)


