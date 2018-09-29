import os
import json
import tensorflow as tf
from annoy import AnnoyIndex
from flask import Flask
from flask import request
from flask import jsonify
from flask import abort
from flask import make_response
from flask import render_template

from utils import GenModelUSE


tf.logging.set_verbosity(tf.logging.WARN)


# Buil the USE model
DATA_PATH = 'data/CMDC/'
UNIQUE_STRINGS_PATH = DATA_PATH + 'all_lines_50.txt.embedded.pkl_unique_strings.csv'

# load the unique lines
with open(UNIQUE_STRINGS_PATH) as f:
  UNIQUE_STRINGS = [line.strip() for line in f]
tf.logging.info('Loaded {} unique embedding strings'.format(len(UNIQUE_STRINGS)))

# define the path of the nearest neighbor model to use
ANNOY_INDEX_PATH = DATA_PATH + 'all_lines_50.txt.ann'

# Load generative models from pickles to generate from scratch.
try:
  tf.logging.info('Build generative model...')
  GEN_MODEL_USE = GenModelUSE(
    annoy_index_path=ANNOY_INDEX_PATH,
    unique_strings=UNIQUE_STRINGS
  )
  tf.logging.info('Generative model built.')
except (OSError, IOError) as e:
  tf.logging.info('Error building generative model.')

# Start the app JANN.
JANN = Flask(__name__)
JANN.config['SECRET_KEY'] = 'IAMASUPERSECRETKEYTHATNOONECANGUESS'

@JANN.errorhandler(404)
def not_found(error):
  """Flask route for 404 errors."""
  return make_response(jsonify({'error': 'Not found'}), 404)

@JANN.route('/')
def index():
    return render_template('index.html')

@JANN.route('/model_inference', methods=['POST', 'GET'])
def model_reply():
  """Flask route to respond to inference request."""
  if 'msg' in request.args:
    message = request.args.get('msg')

    resp = None
    if len(message) > 1:
      try:
        resp = GEN_MODEL_USE.inference(message)
      except Exception as error:
        tf.logging.error('Generative model response error', error)
        resp = None
  else:
      resp = None

  # return the response in a json object
  return json.dumps(resp)

if __name__ == '__main__':
  JANN.run(debug=False, host='0.0.0.0', port=8000, use_reloader=False)
