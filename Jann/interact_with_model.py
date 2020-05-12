import sys
import tensorflow.compat.v1 as tf
import Jann.utils as utils

tf.disable_v2_behavior()


def interact_with_model(args, debug=False):
    """Main run function for interacting with the model."""
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

    if debug:
        return True
    else:
        # build a loop for interactive mode
        while True:
            # get user input
            user_input = input('\nQuery Text: ')
            # if user input is too short
            if len(user_input) < 1:
                continue
            nns, distances = gen_model_use.inference(
                user_input,
                num_neighbors=args.num_neighbors,
                use_sentence_piece=args.use_sentence_piece)

            # print all the returned responses, and distance to input
            for nn, distance in zip(nns, distances):
                print('d: {}, {}'.format(
                    distance,
                    unique_strings[nn].split(args.delimiter)))


if __name__ == '__main__':
    sys.exit(interact_with_model(sys.argv[1:]))
