import sys
import random
import argparse
import tensorflow as tf
import sentencepiece as spm
import tensorflow_hub as hub
from annoy import AnnoyIndex
from utils import MODULE_PATH, process_to_IDs_in_sparse_format


def main(arguments):
  parser = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--verbose', dest='verbose',
                      help="Verbose", action='store_true')
  parser.add_argument('--num_neighbors', type=int,
    help='number of nearest neighbors to return')
  parser.add_argument('--search_k', type=int,
    help='runtime tradeoff between accuracy and speed')
  parser.add_argument('--path_to_text',
    help="path to original text file")

  parser.set_defaults(
    verbose=True,
    num_neighbors=10,
    search_k=100
  )
  args = parser.parse_args(arguments)

  # Reduce logging output.
  if args.verbose:
    tf.logging.set_verbosity(tf.logging.DEBUG)
  else:
    tf.logging.set_verbosity(tf.logging.INFO)

  tf.logging.info('Loading unique strings.')
  unique_strings_path = args.path_to_text + '.embedded.pkl_unique_strings.csv'
  # load the unique lines
  with open(unique_strings_path) as f:
      unique_strings = [line.rstrip() for line in f]

  tf.logging.info('Lodaded {} unique strings'.format(len(unique_strings)))

  # Length of item vector that will be indexed
  nn_forest = AnnoyIndex(512)

  # Reload approximate nearest neighbor forest
  output_path = args.path_to_text + '.ann'
  nn_forest.load(output_path)

  tf.logging.info('Index forest rebuilt {}'.format(output_path))

  # Reload the embedding module
  module = hub.Module(MODULE_PATH, trainable=False)

  # Start the tensorflow session
  with tf.Session() as session:

    # Initialize the variables
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    tf.logging.info('Interactive session is initialized...')

    # spm_path now contains a path to the SentencePiece
    # model stored inside the TF-Hub module
    spm_path = session.run(module(signature="spm_path"))
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_path)

    # build an input placeholder
    input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])

    # build an input / output from the placeholders
    embeddings = module(inputs=dict(
        values=input_placeholder.values,
        indices=input_placeholder.indices,
        dense_shape=input_placeholder.dense_shape
      )
    )

    # build a loop for interactive mode
    while True:
      # get user input
      user_input = [input('\nQuery Text: ')]

      # if user input is too short
      if len(user_input[0]) < 1:
        continue

      # process unencoded lines to values and IDs in sparse format
      values, indices, dense_shape = process_to_IDs_in_sparse_format(sp=sp,
        sentences=user_input)

      # run the session
      line_embeddings = session.run(
        embeddings,
        feed_dict={
          input_placeholder.values: values,
          input_placeholder.indices: indices,
          input_placeholder.dense_shape: dense_shape
        }
      )

      # extract the query vector of interest
      query_vector = line_embeddings[0]

      # get nearest neighbors
      (nns, distances) = nn_forest.get_nns_by_vector(
        query_vector,
        int(args.num_neighbors),
        search_k=1000,
        include_distances=True)

      for i,nn in enumerate(nns):
        tf.logging.info('{}, d: {}, {}'.format(i, round(distances[i], 3), unique_strings[nn]))

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))









