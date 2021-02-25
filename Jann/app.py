import argparse
import random

import flask_monitoringdashboard as dashboard
import tensorflow.compat.v1 as tf  # type: ignore
from flask import Flask, jsonify, make_response, render_template, request
from flask_cors import CORS, cross_origin

import Jann.utils as utils

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.DEBUG)


# Parse the command line arguments
parser = argparse.ArgumentParser(description='Jann server')
parser.add_argument('--port', type=int, default=5000, help='Port of the server.')
parser.add_argument('--neighbors', type=int, default=5,
                    help='Number of neighbors from which to sample from.')
parser.add_argument('--data_key', default='all_lines_50',
                    help='Name of the data_key file.')
parser.add_argument('--data_path', default='data/CMDC/',
                    help='Location of the TF Annoy model.')
opts, unknown = parser.parse_known_args()
print(opts)


# Do we want to return the nearest neighbor? Or sample from several ones?
sample_from_n_neighbors = opts.neighbors

# Buil the USE model
model_name = '{}.txt.embedded.pkl_unique_strings.csv'.format(opts.data_key)
unique_strings_path = (opts.data_path + model_name)

# load the unique lines
with open(unique_strings_path) as f:
    unique_strings = [line.strip() for line in f]
tf.logging.info('Loaded {} unique strings'.format(len(unique_strings)))

# define the path of the nearest neighbor model to use
annoy_index_path = opts.data_path + '{}.txt.ann'.format(opts.data_key)

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

# Start the app JANN, and wrap it in CORS handler
JANN = Flask(__name__)
CORS(JANN, support_credentials=True)
JANN.config['SECRET_KEY'] = 'IAMASUPERSECRETKEYTHATNOONECANGUESS'
dashboard.bind(JANN)


@JANN.errorhandler(404)
@cross_origin(supports_credentials=True)
def not_found(error):
    """Flask route for 404 errors."""
    return make_response(jsonify({'error': 'Not found'}), 404)


@JANN.route('/')
@cross_origin(supports_credentials=True)
def index():
    return render_template('index.html',
                           title='Home')


@JANN.route('/model_inference', methods=['POST', 'GET'])
@cross_origin(supports_credentials=True)
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
        port=opts.port,
        use_reloader=False)
