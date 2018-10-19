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
  parser.add_argument('--infile', help="Input file")
  parser.add_argument('--outfile', help="Output file")
  parser.add_argument('--num_lines', type=int, help="Number of pairs")
  parser.add_argument('--delimiter', default='\t', help="Delimiter")
  parser.add_argument('--verbose', dest='verbose',
                      help="Verbose", action='store_true')
  parser.set_defaults(verbose=True)
  args = parser.parse_args(arguments)

  # Reduce logging output.
  if args.verbose:
    tf.logging.set_verbosity(tf.logging.DEBUG)
  else:
    tf.logging.set_verbosity(tf.logging.WARN)

  tf.logging.log(tf.logging.INFO,
    "Selecting and saving {} random pairs...".format(args.num_lines))
  tf.logging.log(tf.logging.INFO, 'Input file: {}'.format(args.infile))

  # load the lines
  lines, _ = load_data(args.infile, dest_type='list', delimiter='\n')
  tf.logging.log(tf.logging.INFO, "Loaded {} lines: {}".format(len(lines), args.infile))

  with open(args.outfile, 'w', encoding='iso-8859-1') as outputfile:
    writer = csv.writer(outputfile, delimiter=args.delimiter)
    collected_pairs = extract_pairs_from_lines(lines)
    random_idxs = np.random.choice(len(collected_pairs), args.num_lines, replace=False)
    for random_id in random_idxs:
      pair = collected_pairs[random_id]
      writer.writerow(pair)

  tf.logging.log(tf.logging.INFO,
    "Wrote {} pairs to {}.".format(args.num_lines, args.outfile))


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))