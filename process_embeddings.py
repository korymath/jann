from __future__ import print_function

import io
import os
import re
import sys
import pickle
import hashlib
import argparse
import numpy as np
import tensorflow as tf

from utils import *


def main(arguments):
  """Main run code."""
  parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--path_to_text', help="path to original text file")
  parser.add_argument('--pairs', dest='pairs',
                      help="Pairs", action='store_true')
  parser.add_argument('--delimiter', default='\t', help="Verbose")
  parser.add_argument('--verbose', dest='verbose',
                      help="Verbose", action='store_true')
  parser.set_defaults(verbose=True)
  args = parser.parse_args(arguments)

  path_to_embeddings = args.path_to_text + '.embedded.pkl'

  # Reduce logging output.
  if args.verbose:
    tf.logging.set_verbosity(tf.logging.DEBUG)
  else:
    tf.logging.set_verbosity(tf.logging.INFO)

  # load the embeddings data object
  embeddings, _ = load_data(path_to_embeddings, 'dict')
  tf.logging.log(tf.logging.INFO,
    '{} lines in embeddings: {}'.format(len(embeddings.keys()),
      path_to_embeddings))

  all_embeddings = []
  with open(path_to_embeddings + '_unique_strings.csv', 'wb') as outfile:
    for k,v in embeddings.items():
      output_line = v['line'].encode('utf-8')
      if args.pairs:
        output_line_response = v['response'].encode('utf-8')
        outfile.write(output_line + args.delimiter.encode('utf-8') + output_line_response + b'\n')
      else:
        outfile.write(output_line+b'\n')
      all_embeddings.append(v['line_embedding'])

  # Convert to a numpy array
  all_embeddings_np = np.array([np.array(xi) for xi in all_embeddings])
  array_outfile = path_to_embeddings + '_unique_strings_embeddings.txt'
  np.savetxt(array_outfile, all_embeddings_np)

  # Print the embedding shape
  tf.logging.log(tf.logging.INFO,
    'Embedings shape {}'.format(all_embeddings_np.shape))


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
