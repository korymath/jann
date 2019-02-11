import sys
import tensorflow as tf

import utils


def main(arguments):
    """Main run function for interacting with the model."""

    # Parse the arguments
    args = utils.parse_arguments(arguments)

    tf.logging.info('Loading unique strings.')

    data_path = args.infile
    unique_strings_path = data_path + '.embedded.pkl_unique_strings.csv'
    # load the unique lines
    with open(unique_strings_path) as f:
        unique_strings = [line.rstrip() for line in f]

    tf.logging.info('Lodaded {} unique strings'.format(len(unique_strings)))

    # define the path of the nearest neighbor model to use
    annoy_index_path = data_path + '.ann'

    # Load generative models from pickles to generate from scratch.
    try:
        tf.logging.info('Build generative model...')
        gen_model_use = utils.GenModelUSE(
            annoy_index_path=annoy_index_path,
            unique_strings=unique_strings,
            module_path=args.module_path,
            use_sentence_piece=args.use_sentence_piece
        )
        tf.logging.info('Generative model built.')
    except (OSError, IOError) as e:
        tf.logging.error(e)
        tf.logging.info('Error building generative model.')

    # build a loop for interactive mode
    while True:
        # get user input
        user_input = input('\nQuery Text: ')
        # if user input is too short
        if len(user_input) < 1:
            continue
        resp = gen_model_use.inference(user_input, num_neighbors=1,
                                       args=args).split(args.delimiter)
        # respond with the response from the [input,response] pair
        # using matching to the input string
        tf.logging.info('Closest matched root: {}'.format(resp[0]))
        if len(resp) > 0:
            tf.logging.info('Response: {}'.format(resp[0]))


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
