import tensorflow.compat.v1 as tf
from flask import Flask
from flask import request
from flask import jsonify
from flask import make_response
import utils

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.DEBUG)


# Buil the USE model
data_path = 'data/CMDC/'
model_name = 'all_lines_50_pairs.txt.embedded.pkl_unique_strings.csv'
unique_strings_path = (data_path + model_name)

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


@JANN.route('/model_inference', methods=['POST', 'GET'])
def model_reply():
    """Flask route to respond to inference request."""

    # Initialize a blank message and blank response
    message = None
    resp = None

    # Log the request
    tf.logging.info(request)

    # If msg is set in the request arguments, this is direct to URL
    if 'msg' in request.args:
        message = request.args.get('msg')
        tf.logging.info('message from msg argument: {}'.format(message))
    else:
        # This is a dialogflow request, follow the Dialogflow protocol
        data_json = request.get_json(silent=True, force=True)
        print(data_json)
        message = data_json["queryResult"]["queryText"]

    # If the message exists, then use it to generate an inference
    if message:
        try:
            nns, distances = gen_model_use.inference(message)

            # print all the returned responses, and distance to input
            for nn, distance in zip(nns, distances):
                tf.logging.info('d: {}, {}'.format(
                    distance,
                    unique_strings[nn].split('\t')))  # args.delimiter

            # for example we can take the response (index 1)
            # to the closest neighbor
            gen_resp = unique_strings[nns[0]].split('\t')[1]
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
        port=8000,
        use_reloader=False)
