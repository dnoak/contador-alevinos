import sys
from time import time
sys.path.append('../..')
from src.predictor.counter import CounterModel
import dataclasses
from flask import Flask, request
import json
import urllib3
import os
#import ssl

#ssl._create_default_https_context = ssl._create_unverified_context
os.environ["PYTHONHTTPSVERIFY"] = "0"
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

application = Flask(__name__)

backend = CounterModel(
    model_name='detr-resnet-50',
)

@application.route('/')
def server_index():
    return 'API - CONTADOR DE ALEVINOS: ONLINE'

@application.route('/set-params', methods=['POST'])
def post_set_params():
    global backend
    params = request.get_json()
    backend = CounterModel(**params)
    return json.dumps(dataclasses.asdict(backend))

@application.route('/get-params', methods=['GET'])
def get_get_params():
    return json.dumps(dataclasses.asdict(backend))

@application.route('/contador-alevinos', methods=['POST'])
def post_contador_alevinos():
    result = {"results": []}
    post_list = request.get_json()
    for post in post_list:
        result["results"] += [backend.count(**post)]
    return json.dumps(result)

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=3000, debug=False)

