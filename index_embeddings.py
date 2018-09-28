import sys
import random
import argparse
import tensorflow as tf

from annoy import AnnoyIndex


def main(arguments):
  parser = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--verbose', dest='verbose',
                      help="Verbose", action='store_true')
  parser.add_argument('--path_to_text', help="path to original text file")
  parser.add_argument('--num_trees', type=int,
    help='number of trees for approximate nearest neighbor')

  parser.set_defaults(
    verbose=True,
    num_trees=100,
  )
  args = parser.parse_args(arguments)

  # Reduce logging output.
  if args.verbose:
    tf.logging.set_verbosity(tf.logging.DEBUG)
  else:
    tf.logging.set_verbosity(tf.logging.INFO)

  unique_strings_path = args.path_to_text + '.embedded.pkl_unique_strings.csv'
  # load the unique lines
  with open(unique_strings_path) as f:
      unique_strings = [line.rstrip() for line in f]

  unique_embeddings_path = args.path_to_text + '.embedded.pkl_unique_strings_embeddings.txt'
  # load the unique embeddings
  with open(unique_embeddings_path) as f:
      unique_embeddings = [[float(x) for x in line.strip().split()] for line in f]

  tf.logging.info('Lodaded {} unique strings, and {} embeddings of dimension {}'.
    format(len(unique_strings), len(unique_embeddings), len(unique_embeddings[0])))

  # Length of item vector that will be indexed
  nn_forest = AnnoyIndex(512)

  for i in range(len(unique_strings)):
    v = unique_embeddings[i]
    nn_forest.add_item(i, v)

  # Build an approximate nearest neighbor forest with num_trees
  nn_forest.build(int(args.num_trees))
  output_path = args.path_to_text + '.ann'
  nn_forest.save(output_path)

  tf.logging.info('Index forest built {}'.format(output_path))

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))