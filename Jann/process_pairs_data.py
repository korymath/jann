import csv
import sys
import numpy as np
import tensorflow.compat.v1 as tf
import utils

tf.disable_v2_behavior()


def main(arguments):
    """Main run function to process the pairs data."""

    # Parse the arguments
    args = utils.parse_arguments(arguments)

    tf.logging.info('Select and save {} random pairs...'.format(
      args.num_lines))
    tf.logging.info('Input file: {}'.format(args.infile))

    # load the lines
    lines, _ = utils.load_data(args.infile, dest_type='list', delimiter='\n')
    tf.logging.info("Loaded {} lines: {}".format(len(lines), args.infile))

    with open(args.outfile, 'w', encoding='iso-8859-1') as outputfile:
        writer = csv.writer(outputfile, delimiter=args.delimiter)
        collected_pairs = utils.extract_pairs_from_lines(lines)
        random_idxs = np.random.choice(
          len(collected_pairs), args.num_lines, replace=False)
        for random_id in random_idxs:
            pair = collected_pairs[random_id]
            writer.writerow(pair)

    tf.logging.info('Wrote {} pairs to {}.'.format(
      args.num_lines, args.outfile))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
