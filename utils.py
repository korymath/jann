import os
import io
import csv
import random
import pickle
import hashlib
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import sentencepiece as spm
import tensorflow_hub as hub
from annoy import AnnoyIndex


def parse_arguments(arguments=None):
  parser = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('--module_path',
                      help='Specify the local encoder model path',
                      default='data/modules/universal-sentence-encoder-lite-2')
  parser.add_argument('--infile',
                      help="Path to input file.",
                      default='data/CMDC/all_lines_50.txt')
  parser.add_argument('--infile_path',
                      help="Input file path")
  parser.add_argument('--outfile',
                      help="Output file.",
                      default=None)
  parser.add_argument('--num_lines',
                      type=int,
                      help="Number of lines to processes")
  parser.add_argument('--pairs',
                      dest='pairs',
                      help="Flag to use pairs mode or not.",
                      action='store_true',
                      default=False)
  parser.add_argument('--delimiter',
                      default='\t',
                      help="Delimeter between input<>response.")
  parser.add_argument('--verbose',
                      dest='verbose',
                      help="Flag to add verbose logging detail.",
                      action='store_true',
                      default=True)
  parser.add_argument('--num_trees',
                      type=int,
                      help='Number of trees for to search for neighbors.',
                      default=100)
  parser.add_argument('--num_neighbors',
                      type=int,
                      help='Number of nearest neighbors to return.',
                      default=10)
  parser.add_argument('--search_k',
                      type=int,
                      help='Number of trees to search.',
                      default=10)
  parser.add_argument('--use_sentence_piece',
                      type=bool,
                      default=True)
  args = parser.parse_args(arguments)

  # Specify the local module path
  if 'lite' in args.module_path:
    args.use_sentence_piece = True

  # Reduce logging output.
  if args.verbose:
    tf.logging.set_verbosity(tf.logging.DEBUG)
  else:
    tf.logging.set_verbosity(tf.logging.WARN)
  return args


def load_data(file_path, dest_type, pairs=False, delimiter='\t'):
  """Load line separated text files into list. """

  if dest_type == 'list':
    if not pairs:
      tempfile = io.open(file_path, 'r', encoding="iso-8859-1", errors='ignore')
      dest = []
      for line in tempfile:
        clean_string = line.strip()
        # check if blank
        if clean_string:
          dest.append(clean_string)
      tempfile.close()
      dest2 = None
    else:
      first_lines = []
      second_lines = []
      tf.logging.info('Loading pairs data')
      with open(file_path, 'r', encoding='iso-8859-1') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
          first_lines.append(row[0])
          second_lines.append(row[1])
      dest = first_lines
      dest2 = second_lines
  elif dest_type == 'dict':
    with open(file_path, 'rb') as f:
      dest = pickle.load(f)
    dest2 = None
  else:
    dest = None
    dest2 = None
    tf.logging.info('Bad destination data type specified.')
  return dest, dest2


def load_lines(fname, fields):
  """Load Cornell Movie Dialog Lines."""
  lines = {}
  with open(fname, 'r', encoding='iso-8859-1') as f:
    for line in f:
      values = line.split(" +++$+++ ")
      line_obj = {}
      for i, field in enumerate(fields):
        line_obj[field] = values[i]
      lines[line_obj['lineID']] = line_obj
    return lines


def load_conversations(fname, lines, fields):
  """Load Cornell Movie Dialog Conversations."""
  convos = []
  with open(fname, 'r', encoding='iso-8859-1') as f:
    for line in f:
      values = line.split(" +++$+++ ")
      conv_obj = {}
      for i, field in enumerate(fields):
          conv_obj[field] = values[i]
      # Convert string to list
      line_ids = eval(conv_obj["utteranceIDs"])
      conv_obj["lines"] = []
      for line_id in line_ids:
        conv_obj["lines"].append(lines[line_id])
      convos.append(conv_obj)
    return convos


def extract_pairs(conversations):
  """Extract pairs from the Cornell Movie Dialog Conversations."""
  collected_pairs = []
  for conversation in conversations:
    # ignore last line
    for i in range(len(conversation["lines"]) - 1):
      first_line = conversation["lines"][i]["text"].strip()
      second_line = conversation["lines"][i+1]["text"].strip()
      if first_line and second_line:
        collected_pairs.append([first_line, second_line])
  return collected_pairs


def extract_pairs_from_lines(lines):
  """Extract pairs from Cornell Movie Dialog Lines."""
  collected_pairs = []
  for i in range(len(lines) - 1):
    first_line = lines[i].strip()
    second_line = lines[i+1].strip()
    if first_line and second_line:
      collected_pairs.append([first_line, second_line])
  return collected_pairs


def process_to_IDs_in_sparse_format(sp, sentences):
  # A utility method that processes sentences with the sentence piece processor
  # 'sp' and returns the results in tf.SparseTensor-similar format:
  # (values, indices, dense_shape)
  ids = [sp.EncodeAsIds(x) for x in sentences]
  max_len = max(len(x) for x in ids)
  dense_shape=(len(ids), max_len)
  values=[item for sublist in ids for item in sublist]
  indices=[[row,col] for row in range(len(ids)) for col in range(len(ids[row]))]
  return (values, indices, dense_shape)


def get_id_chunks(the_big_list, n_sub_list):
  """Yield successive n_sub_list-sized chunks from the_big_list."""
  for i in range(0, len(the_big_list), n_sub_list):
    yield the_big_list[i:i + n_sub_list]


def embed_lines(args, unencoded_lines, output_dict,
                unencoded_lines_responses=None):
  """Embed a collection of lines to an output dictionary."""

  # Import the Universal Sentence Encoder's TF Hub module
  module = hub.Module(args.module_path, trainable=False)
  config = tf.ConfigProto(allow_soft_placement = True)

  with tf.Session(config = config) as session:
    # initialize the variables
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])

    if args.use_sentence_piece:
      # spm_path now contains a path to the SentencePiece
      # model stored inside the TF-Hub module
      spm_path = session.run(module(signature="spm_path"))
      sp = spm.SentencePieceProcessor()
      sp.Load(spm_path)

      # build an input placeholder
      with tf.device('/gpu:0'):
        input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
        embeddings = module(inputs=dict(
          values=input_placeholder.values,
          indices=input_placeholder.indices,
          dense_shape=input_placeholder.dense_shape
          )
        )

    # size of chunk is how many lines will be encoded
    # with each pass of the model
    size_of_chunk = 256

    # ensure that every line has a response
    assert len(unencoded_lines) == len(unencoded_lines_responses)
    all_id_chunks = get_id_chunks(range(len(unencoded_lines)), size_of_chunk)

    for id_chunk in tqdm(all_id_chunks,
      total=(len(unencoded_lines) // size_of_chunk)):

      # get the chunk of lines and matching responses by list of ids
      chunk_unencoded_lines = [unencoded_lines[x] for x in id_chunk]
      chunck_unenc_resp = [unencoded_lines_responses[x] for x in id_chunk]

      if args.use_sentence_piece:
        # process unencoded lines to values and IDs in sparse format
        values, indices, dense_shape = process_to_IDs_in_sparse_format(sp=sp,
          sentences=chunk_unencoded_lines)

        # run the session
        chunk_line_embds = session.run(
          embeddings,
          feed_dict={
            input_placeholder.values: values,
            input_placeholder.indices: indices,
            input_placeholder.dense_shape: dense_shape
          }
        )
      else:
        with tf.device('/gpu:0'):
          chunk_line_embds = session.run(module(chunk_unencoded_lines))

      # hash the object into the full output dataframe
      for i, line_embedding in enumerate(np.array(chunk_line_embds).tolist()):
        if args.verbose:
          tf.logging.info(
            "Line: {}".format(chunk_unencoded_lines[i]))
          tf.logging.info(
            "Embedding size: {}".format(len(line_embedding)))
          snippet = ", ".join((str(x) for x in line_embedding[:3]))
          tf.logging.info(
            "Embedding: [{}, ...]\n".format(snippet))

        # Encode a hash for the string
        hash_object = hashlib.md5(chunk_unencoded_lines[i].encode('utf-8'))

        # Add a row to the dataframe
        output_dict[hash_object.hexdigest()] = {
          'line': chunk_unencoded_lines[i],
          'line_embedding': line_embedding,
          'response': chunck_unenc_resp[i]
        }
  return output_dict

class GenModelUSE(object):
    def __init__(self, annoy_index_path, unique_strings,
                 use_sentence_piece, module_path):
        self.annoy_index_path = annoy_index_path
        self.unique_strings = unique_strings

        # load the annoy index for mmap speed
        # Length of item vector that will be indexed
        self.annoy_index = AnnoyIndex(512)

        # super fast, will just mmap the file
        self.annoy_index.load(self.annoy_index_path)

        g = tf.Graph()
        with g.as_default():
          # define the module
          module = hub.Module(module_path, trainable=False)

          if use_sentence_piece:
            # build an input placeholder
            self.input_placeholder = tf.sparse_placeholder(tf.int64,
                                                           shape=[None, None])
            # build an input / output from the placeholders
            self.embeddings = module(inputs=dict(
              values=self.input_placeholder.values,
              indices=self.input_placeholder.indices,
              dense_shape=self.input_placeholder.dense_shape
              )
            )
          else:
            # build an input placeholder
            self.input_placeholder = tf.placeholder(tf.string, shape=(None))
            self.embeddings = module(self.input_placeholder)

          init_op = tf.group(
            [tf.global_variables_initializer(), tf.tables_initializer()])

        # do not finalize the graph if using sentence piece module
        if not use_sentence_piece:
          g.finalize()

        # define the configuration
        config = tf.ConfigProto(allow_soft_placement = True)
        self.sess = tf.Session(graph=g, config=config)
        self.sess.run(init_op)

        if use_sentence_piece:
          # spm_path now contains a path to the SentencePiece
          # model stored inside the TF-Hub module
          with g.as_default():
            spm_path = self.sess.run(module(signature="spm_path"))
          self.sp = spm.SentencePieceProcessor()
          self.sp.Load(spm_path)

        tf.logging.info('Interactive session is initialized...')

    def inference(self, input_text, num_neighbors=10):
        """Inference from nearest neighbor model."""

        # Handle the short input
        if len(input_text) < 1:
          return 'Say something!'

        tf.logging.info('Input text: {}'.format(input_text))

        # Build a list of the user input
        user_input = [input_text]

        if args.use_sentence_piece:
          # process unencoded lines to values and IDs in sparse format
          values, indices, dense_shape = process_to_IDs_in_sparse_format(
            sp=self.sp, sentences=user_input)

          # run the session
          # Get embedding of the input text
          embeddings = self.sess.run(
            self.embeddings,
            feed_dict={
              self.input_placeholder.values: values,
              self.input_placeholder.indices: indices,
              self.input_placeholder.dense_shape: dense_shape
            }
          )
        else:
          embeddings = self.sess.run(
            self.embeddings,
            feed_dict={
              self.input_placeholder: user_input
            }
          )

        tf.logging.info(
          'Successfully generated {} embeddings of length {}.'.format(
            len(embeddings), len(embeddings[0])))

        # Extract the query vector of interest.
        query_vector = embeddings[0]

        # Get nearest neighbors
        nns = self.annoy_index.get_nns_by_vector(query_vector, num_neighbors,
            search_k=-1, include_distances=False)
        tf.logging.info('Nearest neighbor IDS: {}'.format(nns))
        # tf.logging.info(['{}'.format(
          # self.unique_strings[x].split('\t')) for x in nns])

        # Randomly sample from the top-N nearest neighbors to avoid repetition
        generative_response = self.unique_strings[random.choice(nns)]

        return generative_response
