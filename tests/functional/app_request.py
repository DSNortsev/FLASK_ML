from service import app
from utils import read_json
import json

data = read_json('tests/data/data.json')

responds = app.test_client().post('/predict',
                                  data="test",
                                  content_type='application/json')
print(responds.status_code)
print(json.loads(responds.data))
print(responds.headers['Content-Type'])

