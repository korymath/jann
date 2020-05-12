import sys
import tensorflow.compat.v1 as tf
from annoy import AnnoyIndex
import Jann.utils as utils

tf.disable_v2_behavior()


def index_embeddings(args):
    """Main run function for indexing the embeddings."""
    unique_strings_path = args.infile + '.embedded.pkl_unique_strings.csv'

    # Load the unique lines
    with open(unique_strings_path) as f:
        unique_strings = [line.rstrip() for line in f]

    unique_embeddings_path = (args.infile +
                              '.embedded.pkl_unique_strings_embeddings.txt')
    # Load the unique embeddings
    with open(unique_embeddings_path) as f:
        unique_embeddings = [[float(x) for x in
                              line.strip().split()] for line in f]

    tf.logging.info('Loaded {} unique strings, {} embeddings of dimension {}'.
                    format(len(unique_strings),
                           len(unique_embeddings),
                           len(unique_embeddings[0])))

    # Length of item vector that will be indexed
    nn_forest = AnnoyIndex(512, metric='angular')

    for i in range(len(unique_strings)):
        v = unique_embeddings[i]
        nn_forest.add_item(i, v)

    # Build an approximate nearest neighbor forest with num_trees
    nn_forest.build(int(args.num_trees))
    output_path = args.infile + '.ann'
    nn_forest.save(output_path)

    tf.logging.info('Index forest built {}'.format(output_path))

    return True


if __name__ == '__main__':
    # Parse the arguments
    args = utils.parse_arguments(sys.argv[1:])
    sys.exit(index_embeddings(args))
