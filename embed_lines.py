import os
import sys
import tensorflow as tf

import utils


def main(arguments):
  """Main run function for embed lines."""

  # Parse the arguments
  args = utils.parse_arguments(arguments)

  # Build the input message list
  try:
    lines, response_lines = utils.load_data(args.infile, 'list', args.pairs)
  except FileNotFoundError as e:
    tf.logging.log(tf.logging.ERROR, e)
    sys.exit(0)
  tf.logging.log(tf.logging.INFO,
      '{} lines in input file: {}'.format(len(lines), args.infile))
  if response_lines:
    tf.logging.log(tf.logging.INFO,
        '{} response lines in input file: {}'.format(len(response_lines),
                                                     args.infile))

  # check if the file exists
  output_file_path = args.infile + '.embedded.pkl'
  if os.path.isfile(output_file_path):
    # if it does, load it in
    tf.logging.log(tf.logging.INFO,
        'Loading existing saved output file: {}'.format(output_file_path))
    output_dict = utils.load_obj(output_file_path)

    # Exclude lines which have already been encoded
    unencoded_lines = []
    unencoded_lines_responses = []
    for i, line in enumerate(lines):
      line_hash = utils.hashlib.md5(line.encode('utf-8')).hexdigest()
      if line_hash not in output_dict.keys():
        unencoded_lines.append(line)
        if response_lines == None:
          unencoded_lines_responses.append(lines[i])
        else:
          unencoded_lines_responses.append(response_lines[i])
  else:
    # make a new dataframe
    tf.logging.log(tf.logging.INFO, 'Creating new dictionary to save outputs')
    output_dict = {}
    unencoded_lines = lines
    if response_lines == None:
      unencoded_lines_responses = lines
    else:
      unencoded_lines_responses = response_lines

  tf.logging.log(tf.logging.INFO,
    '{} new lines to encode...'.format(len(unencoded_lines)))

  if len(unencoded_lines) > 0:
    output_dict = utils.embed_lines(args, unencoded_lines,
      output_dict, unencoded_lines_responses)

    # Save output dataframe to pickle
    utils.save_obj(output_dict, output_file_path)
    tf.logging.log(tf.logging.INFO,
      '{} lines embedded and saved. Quitting.'.format(len(output_dict)))
  else:
    tf.logging.log(tf.logging.INFO, 'No new lines encoded. Quitting.')

  # Print the output file path to end
  tf.logging.info('Output file: {}'.format(output_file_path))

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))