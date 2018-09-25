import json
import tensorflow as tf
from annoy import AnnoyIndex
from flask import Flask, request, jsonify, abort, make_response

from utils import GenModelUSE

tf.logging.set_verbosity(tf.logging.DEBUG)

# Buil the USE model
DATA_PATH = 'data/CMDC/'
unique_strings_path = DATA_PATH + 'all_lines_50.txt.embedded.pkl_unique_strings.csv'

# load the unique lines
with open(unique_strings_path) as f:
  unique_strings = [line.strip() for line in f]
tf.logging.info('Loaded {} unique embedding strings'.format(len(unique_strings)))

# define the path of the nearest neighbor model to use
annoy_index_path = DATA_PATH + 'all_lines_50.txt.ann'

# Load generative models from pickles to generate from scratch.
try:
    tf.logging.info('Build GEN_MODEL0_USE...')
    GEN_MODEL_USE = GenModelUSE(
        annoy_index_path=annoy_index_path,
        unique_strings=unique_strings
    )
    tf.logging.info('Generative <model></model> built.')
except (OSError, IOError) as e:
    tf.logging.info('Error building generative model.')

# Start the models app.
app = Flask(__name__)

@app.errorhandler(404)
def not_found(error):
    """Flask route for 404 errors."""
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/model_inference')
def model_reply():
    """Flask route to respond to inference request."""
    if 'msg' in request.args:
        message = request.args.get('msg')
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
    app.run(debug=False, host='0.0.0.0',
        port=5000, use_reloader=False)