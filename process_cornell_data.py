import os
import csv
import sys
import numpy as np
import tensorflow as tf

import utils


def main(arguments):
    """Main run function for processing the Cornell Movie Dialog Data."""

    # Parse the arguments
    args = utils.parse_arguments(arguments)

    # movie lines file
    movie_lines_file = os.path.join(args.infile_path, 'movie_lines.txt')
    # movie conversations file
    movie_conversations_file = os.path.join(args.infile_path,
                                            'movie_conversations.txt')

    if not args.pairs:
        tf.logging.info(
            "Selecting and saving {} random lines...".format(args.num_lines))
        lines = []
        try:
            with open(movie_lines_file, encoding='iso-8859-1') as f:
                for line in f:
                    values = line.split(" +++$+++ ")
                    lines.append(values[-1].strip())
        except FileNotFoundError as error:
            tf.logging.error(error)
            tf.logging.error(tf.logging.ERROR,
                             'File not found.')
            sys.exit(0)
        tf.logging.info("Found {} input lines.".format(len(lines)))

        with open(args.outfile, 'w', encoding='iso-8859-1') as f:
            if args.num_lines != 0:
                for item in np.random.choice(
                  lines, args.num_lines, replace=False):
                    f.write("%s\n" % item)
            else:
                for item in lines:
                    f.write("%s\n" % item)
            tf.logging.info(
              'Wrote {} lines to {}.'.format(
                args.num_lines, args.outfile))
    else:
        tf.logging.info(
          "Selecting and saving {} random pairs...".format(args.num_lines))
        tf.logging.info(
          'CMDC movie_lines_path: {}'.format(movie_lines_file))
        tf.logging.info(
          'CMDC movie_converstions_path: {}'.format(movie_conversations_file))

        movie_lines_fields = ["lineID", "characterID",
                              "movieID", "character", "text"]
        movie_conversations_fields = ["character1ID", "character2ID",
                                      "movieID", "utteranceIDs"]

        # load the lines
        lines = utils.load_lines(movie_lines_file, movie_lines_fields)
        tf.logging.info(
            "Loaded {} lines: {}".format(len(lines), movie_lines_file))

        # load the conversations
        conversations = utils.load_conversations(
          movie_conversations_file, lines, movie_conversations_fields)
        tf.logging.info("Loaded {} conversations: {}".format(
          len(conversations), movie_conversations_file))

        with open(args.outfile, 'w', encoding='iso-8859-1') as outputfile:
            writer = csv.writer(outputfile, delimiter=args.delimiter)
            collected_pairs = utils.extract_pairs(conversations)
            tf.logging.info(
                'Total of {} pairs'.format(len(collected_pairs)))
            if int(args.num_lines) != 0:
                random_idxs = np.random.choice(
                  len(collected_pairs), args.num_lines, replace=False)
                for random_id in random_idxs:
                    pair = collected_pairs[random_id]
                    writer.writerow(pair)
                tf.logging.info("Wrote {} pairs to {}.".format(
                    args.num_lines, args.outfile))
            else:
                for item in collected_pairs:
                    writer.writerow(item)
                tf.logging.info("Wrote {} pairs to {}.".format(
                    len(collected_pairs), args.outfile))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
