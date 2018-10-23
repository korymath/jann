import os
import csv
import sys
import argparse
import numpy as np
import tensorflow as tf

from utils import *


def main(arguments):
  parser = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--infile_path', help="Input file path")
  parser.add_argument('--outfile', help="Verbose")
  parser.add_argument('--num_lines', type=int, help="Verbose")
  parser.add_argument('--delimiter', default='\t', help="Verbose")
  parser.add_argument('--pairs', dest='pairs',
                      help="Pairs", action='store_true')
  parser.add_argument('--verbose', dest='verbose',
                      help="Verbose", action='store_true')
  parser.set_defaults(verbose=True)
  args = parser.parse_args(arguments)

  # Reduce logging output.
  if args.verbose:
    tf.logging.set_verbosity(tf.logging.DEBUG)
  else:
    tf.logging.set_verbosity(tf.logging.WARN)

  # movie lines file
  movie_lines_file = os.path.join(args.infile_path, 'movie_lines.txt')
  # movie conversations file
  movie_conversations_file = os.path.join(args.infile_path,'movie_conversations.txt')

  if not args.pairs:
    tf.logging.log(tf.logging.INFO,
      "Selecting and saving {} random lines...".format(args.num_lines))
    lines = []
    try:
      with open(movie_lines_file, encoding='iso-8859-1') as f:
        for line in f:
          values = line.split(" +++$+++ ")
          lines.append(values[-1].strip())
    except FileNotFoundError as error:
      tf.logging.log(tf.logging.ERROR, error)
      tf.logging.log(tf.logging.ERROR, 'Input file not found, correct the specified location.')
      sys.exit(0)
    tf.logging.log(tf.logging.INFO, "Found {} input lines.".format(len(lines)))

    with open(args.outfile, 'w', encoding='iso-8859-1') as f:
      if args.num_lines != 0:
        for item in np.random.choice(lines, args.num_lines, replace=False):
          f.write("%s\n" % item)
      else:
        for item in lines:
          f.write("%s\n" % item)
    tf.logging.log(tf.logging.INFO, "Wrote {} lines to {}.".format(len(lines), args.outfile))
  else:
      tf.logging.log(tf.logging.INFO,
        "Selecting and saving {} random pairs...".format(args.num_lines))
      tf.logging.log(tf.logging.INFO, 'CMDC movie_lines_path: {}'.format(movie_lines_file))
      tf.logging.log(tf.logging.INFO, 'CMDC movie_converstions_path: {}'.format(movie_conversations_file))

      movie_lines_fields = ["lineID", "characterID", "movieID", "character", "text"]
      movie_conversations_fields = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

      # load the lines
      lines = load_lines(movie_lines_file, movie_lines_fields)
      tf.logging.log(tf.logging.INFO, "Loaded {} lines: {}".format(len(lines), movie_lines_file))

      # load the conversations
      conversations = load_conversations(movie_conversations_file, lines, movie_conversations_fields)
      tf.logging.log(tf.logging.INFO, "Loaded {} conversations: {}".format(len(conversations), movie_conversations_file))

      with open(args.outfile, 'w', encoding='iso-8859-1') as outputfile:
        writer = csv.writer(outputfile, delimiter=args.delimiter)
        collected_pairs = extract_pairs(conversations)
        print('Total of {} pairs'.format(len(collected_pairs)))
        if int(args.num_lines) != 0:
          random_idxs = np.random.choice(len(collected_pairs), args.num_lines, replace=False)
          for random_id in random_idxs:
            pair = collected_pairs[random_id]
            writer.writerow(pair)
        else:
          for item in collected_pairs:
            writer.writerow(item)

      tf.logging.log(tf.logging.INFO,
        "Wrote {} pairs to {}.".format(args.num_lines, args.outfile))


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
