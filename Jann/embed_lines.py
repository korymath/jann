import os
import sys
import pickle
import tensorflow.compat.v1 as tf
import utils

tf.disable_v2_behavior()


def main(arguments):
    """Main run function for embed lines."""

    # Parse the arguments
    args = utils.parse_arguments(arguments)

    # Build the input message list
    try:
        lines, response_lines = utils.load_data(
          args.infile, 'list', args.pairs)
    except FileNotFoundError as e:
        tf.compat.v1.logging.log(tf.compat.v1.logging.ERROR, e)
        sys.exit(0)
    tf.compat.v1.logging.info('{} lines in input file: {}'.format(
      len(lines), args.infile))
    if response_lines:
        tf.compat.v1.logging.info(
          '{} response lines in input file: {}'.format(
            len(response_lines), args.infile))

    # check if the file exists
    output_file_path = args.infile + '.embedded.pkl'
    if os.path.isfile(output_file_path):
        # if it does, load it in
        tf.compat.v1.logging.info(
          'Loading existing saved output file: {}'.format(
            output_file_path))
        with open(output_file_path, 'rb') as f:
            output_dict = pickle.load(f)

        # Exclude lines which have already been encoded
        unencoded_lines = []
        unencoded_lines_responses = []
        for i, line in enumerate(lines):
            line_hash = utils.hashlib.md5(line.encode('utf-8')).hexdigest()
        if line_hash not in output_dict.keys():
            unencoded_lines.append(line)
            if not response_lines:
                unencoded_lines_responses.append(lines[i])
            else:
                unencoded_lines_responses.append(response_lines[i])
    else:
        # make a new dataframe
        tf.compat.v1.logging.info(
            'Creating new dictionary to save outputs')
        output_dict = {}
        unencoded_lines = lines
        if not response_lines:
            unencoded_lines_responses = lines
        else:
            unencoded_lines_responses = response_lines

    tf.compat.v1.logging.info(
        '{} new lines to encode...'.format(len(unencoded_lines)))

    if len(unencoded_lines) > 0:
        output_dict = utils.embed_lines(args,
                                        unencoded_lines,
                                        output_dict,
                                        unencoded_lines_responses)

        # Save output dataframe to pickle
        with open(output_file_path, 'wb') as f:
            pickle.dump(output_dict, f, pickle.HIGHEST_PROTOCOL)

        tf.compat.v1.logging.info(
            '{} lines embedded and saved. Quitting.'.format(
                len(output_dict)))
    else:
        tf.compat.v1.logging.info('No new lines encoded. Quitting.')

    # Print the output file path to end
    tf.compat.v1.logging.info('Output file: {}'.format(output_file_path))


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
