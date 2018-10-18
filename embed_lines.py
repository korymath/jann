import os
import sys
import csv
import argparse
import tensorflow as tf

from utils import *


def main(arguments):
  """Main run code."""

  parser = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('infile', help="Input file")
  parser.add_argument('--pairs', dest='pairs',
                      help="Pairs", action='store_true')
  parser.add_argument('--verbose', dest='verbose',
                      help="Verbose", action='store_true')
  parser.set_defaults(verbose=False, pairs=False)
  args = parser.parse_args(arguments)

  # Reduce logging output.
  if args.verbose:
    tf.logging.set_verbosity(tf.logging.DEBUG)
  else:
    tf.logging.set_verbosity(tf.logging.WARN)

  # Build the input message list
  lines = load_data(args.infile, 'list', args.pairs)
  tf.logging.log(tf.logging.INFO,
    '{} lines in input file: {}'.format(len(lines), args.infile))

  # check if the file exists
  output_file_path = args.infile + '.embedded.pkl'
  if os.path.isfile(output_file_path):
    # if it does, load it in
    tf.logging.log(tf.logging.INFO,
      'Loading existing saved output file: {}'.format(output_file_path))
    output_dict = load_obj(output_file_path)

    # Remove lines which have already been encoded
    unencoded_lines = []
    for line in lines:
      line_hash = hashlib.md5(line.encode('utf-8')).hexdigest()
      if line_hash not in output_dict.keys():
        unencoded_lines.append(line)
  else:
    # make a new dataframe
    tf.logging.log(tf.logging.INFO, 'Creating new dictionary to save outputs')
    output_dict = {}
    unencoded_lines = lines

  tf.logging.log(tf.logging.INFO,
    '{} new lines to encode...'.format(len(unencoded_lines)))

  if len(unencoded_lines) > 0:
    output_dict = embed_lines(args, unencoded_lines, output_dict)
    # Save output dataframe to pickle
    save_obj(output_dict, output_file_path)
    tf.logging.log(tf.logging.INFO,
      '{} lines embedded and saved. Quitting.'.format(len(output_dict)))
  else:
    tf.logging.log(tf.logging.INFO, 'No new lines encoded. Quitting.')

  # Print the output file path to end
  tf.logging.info('Output file: {}'.format(output_file_path))

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
