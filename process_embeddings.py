import sys
import numpy as np
import tensorflow as tf

import utils


def main(arguments):
    """Main run code for processing embeddings."""

    # Parse the arguments
    args = utils.parse_arguments(arguments)

    # load the embeddings data object
    path_to_embeddings = args.infile + '.embedded.pkl'
    embeddings, _ = utils.load_data(path_to_embeddings, 'dict')
    tf.logging.info(
      '{} lines in embeddings: {}'.format(len(embeddings.keys()),
                                          path_to_embeddings))

    all_embeddings = []
    with open(path_to_embeddings + '_unique_strings.csv', 'wb') as outfile:
        for k, v in embeddings.items():
            output_line = v['line'].encode('utf-8')
            if args.pairs:
                output_line_response = v['response'].encode('utf-8')
                outfile.write(output_line +
                              args.delimiter.encode('utf-8') +
                              output_line_response + b'\n')
        else:
            outfile.write(output_line+b'\n')
        all_embeddings.append(v['line_embedding'])

    # Convert to a numpy array
    all_embeddings_np = np.array([np.array(xi) for xi in all_embeddings])
    array_outfile = path_to_embeddings + '_unique_strings_embeddings.txt'
    np.savetxt(array_outfile, all_embeddings_np)

    # Log the embedding shape
    tf.logging.info(
      'Embedings shape {}'.format(all_embeddings_np.shape))


if __name__ == '__main__':
    main(sys.argv[1:])
