import random

import flask_monitoringdashboard as dashboard
import tensorflow.compat.v1 as tf  # type: ignore
from flask import Flask, jsonify, make_response, render_template, request

import Jann.utils as utils

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.DEBUG)

# Dataset-specific paths
num_samples = 0
data_key = 'all_lines_{}_pairs'.format(num_samples)
data_path = 'data/CMDC/'
# data_key = 'all_lines_10000'
# data_path = 'data/novel-first-lines-dataset/'

# Do we want to return the nearest neighbor?
# sample_from_n_neighbors = 1
# or sample from the nearest N neighbors?
sample_from_n_neighbors = 5

# Buil the USE model
model_name = '{}.txt.embedded.pkl_unique_strings.csv'.format(
    data_key)
unique_strings_path = (data_path + model_name)

# load the unique lines
with open(unique_strings_path) as f:
    unique_strings = [line.strip() for line in f]
tf.logging.info('Loaded {} unique strings'.format(len(unique_strings)))

# define the path of the nearest neighbor model to use
annoy_index_path = data_path + '{}.txt.ann'.format(data_key)

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
dashboard.bind(JANN)


@JANN.errorhandler(404)
def not_found(error):
    """Flask route for 404 errors."""
    return make_response(jsonify({'error': 'Not found'}), 404)


@JANN.route('/')
def index():
    return render_template('index.html',
                           title='Home')


@JANN.route('/model_inference', methods=['POST', 'GET'])
def model_reply():
    """Flask route to respond to inference request."""

    # Initialize a blank message and blank response
    message = None
    resp = None
    return_all_samples = False

    # Log the request
    tf.logging.info(request)

    # If msg is set in the request arguments, this is direct to URL
    if 'msg' in request.args:
        message = request.args.get('msg')
        tf.logging.info('message from msg argument: {}'.format(message))
    else:
        if 'msgs' in request.args:
            message = request.args.get('msgs')
            tf.logging.info('message from msgs argument: {}'.format(message))
            return_all_samples = True
        else:
            # This is a dialogflow request, follow the Dialogflow protocol
            data_json = request.get_json(silent=False, force=True)
            tf.logging.debug('JSON Data: {}'.format(data_json))
            try:
                message = data_json["queryResult"]["queryText"]
            except TypeError as e:
                tf.logging.debug(e)

    # If the message exists, then use it to generate an inference
    if message:
        try:
            nns, distances = gen_model_use.inference(message)
            # print all the returned responses, and distance to input
            for nn, distance in zip(nns, distances):
                tf.logging.info('d: {}, {}'.format(
                    distance,
                    unique_strings[nn].split('\t')))  # args.delimiter

            if return_all_samples:
                # Return all the closest N neighbors
                neighbor_sample = nns[:sample_from_n_neighbors]
            else:
                # Sample from the closest N neighbors
                neighbor_sample = [random.choice(nns[:sample_from_n_neighbors])]

            # recall the neighbor is index 0 and the response is index 1
            gen_resp = '\t'.join([unique_strings[idx].split('\t')[1]
                                  for idx in neighbor_sample])

            # Build the response as the fulfillment text
            resp = {'fulfillmentText': gen_resp}

        except Exception as error:
            tf.logging.error('Generative model response error', error)
            resp = {'fulfillmentText': 'None'}
    else:
        tf.logging.error('No message error')
        resp = {'fulfillmentText': 'None'}

    # return the response in a json object
    return jsonify(resp)


if __name__ == '__main__':
    # Parse the default arguments
    JANN.config['TRAP_BAD_REQUEST_ERRORS'] = True
    JANN.run(
        debug=False,
        host='0.0.0.0',
        use_reloader=False)
