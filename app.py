import json
import tensorflow as tf
from flask import Flask
from flask import request
from flask import jsonify
from flask import make_response
from flask import render_template

import utils

tf.logging.set_verbosity(tf.logging.WARN)


# Buil the USE model
data_path = 'data/CMDC/'
unique_strings_path = (data_path +
                       'all_lines_50.txt.embedded.pkl_unique_strings.csv')

# load the unique lines
with open(unique_strings_path) as f:
    unique_strings = [line.strip() for line in f]
tf.logging.info('Loaded {} unique strings'.format(len(unique_strings)))

# define the path of the nearest neighbor model to use
annoy_index_path = data_path + 'all_lines_50.txt.ann'

# Load generative models from pickles to generate from scratch.
try:
    tf.logging.info('Build generative model...')
    gen_model_use = utils.GenModelUSE(
        annoy_index_path=annoy_index_path,
        unique_strings=unique_strings,
        module_path='data/module/universal-sentence-encoder-lite-2',
        use_sentence_piece=True
    )
    tf.logging.info('Generative model built.')
except (OSError, IOError) as e:
    tf.logging.error(e)
    tf.logging.error('Error building generative model.')

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
        resp = None

    # direct testing through web interface
    message = request.args.get('msg')
    if len(message) > 1:
        try:
            resp = gen_model_use.inference(message)
        except Exception as error:
            tf.logging.error('Generative model response error', error)
            resp = None
    else:
        # dialogflow request
        data_json = request.get_json(silent=True, force=True)
        message = data_json["queryResult"]["queryText"]

    if len(message) > 1:
        try:
            gen_resp = gen_model_use.inference(message)
            resp = {'fulfillmentText': gen_resp}
        except Exception as error:
            tf.logging.error('Generative model response error', error)
            resp = {'fulfillmentText': 'None'}

    # return the response in a json object
    return json.dumps(resp)


if __name__ == '__main__':
    JANN.run(debug=False, host='0.0.0.0', port=8000, use_reloader=True)
