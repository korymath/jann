import os
import io
import pickle
import hashlib
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import sentencepiece as spm
import tensorflow_hub as hub

# Specify the local module path
MODULE_PATH = 'https://tfhub.dev/google/universal-sentence-encoder-lite/2'


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def load_data(file_path, dest_type):
  """Load line separated text files into list. """
  if dest_type == 'list':
    tempfile = io.open(file_path, 'r', encoding="utf-8", errors='ignore')
    dest = []
    for line in tempfile:
      clean_string = line.strip()
      # check if blank
      if clean_string:
        dest.append(clean_string)
    tempfile.close()
  elif dest_type == 'dict':
    dest = load_obj(file_path)
  else:
    dest = None
    print('Bad destination data type specified.')
  return dest

def process_to_IDs_in_sparse_format(sp, sentences):
  # An utility method that processes sentences with the sentence piece processor
  # 'sp' and returns the results in tf.SparseTensor-similar format:
  # (values, indices, dense_shape)
  ids = [sp.EncodeAsIds(x) for x in sentences]
  max_len = max(len(x) for x in ids)
  dense_shape=(len(ids), max_len)
  values=[item for sublist in ids for item in sublist]
  indices=[[row,col] for row in range(len(ids)) for col in range(len(ids[row]))]
  return (values, indices, dense_shape)

def chunks(the_big_list, n_sub_list):
  """Yield successive n_sub_list-sized chunks from the_big_list."""
  for i in range(0, len(the_big_list), n_sub_list):
    yield the_big_list[i:i + n_sub_list]

def embed_lines(args, unencoded_lines, output_dict):
  # Import the Universal Sentence Encoder's TF Hub module

  module = hub.Module(MODULE_PATH, trainable=False)

  with tf.Session() as session:
    # initialize the variables
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])

    # spm_path now contains a path to the SentencePiece
    # model stored inside the TF-Hub module
    spm_path = session.run(module(signature="spm_path"))
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_path)

    # build an input placeholder
    input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
    embeddings = module(
      inputs=dict(
        values=input_placeholder.values,
        indices=input_placeholder.indices,
        dense_shape=input_placeholder.dense_shape
      )
    )

    # size of chunk is how many lines will be encoded
    # with each pass of the model
    size_of_chunk = 64
    all_chunks = chunks(unencoded_lines, size_of_chunk)

    for chunk_unencoded_lines in tqdm(all_chunks, total=(len(unencoded_lines) // size_of_chunk)):
      # process unencoded lines to values and IDs in sparse format
      values, indices, dense_shape = process_to_IDs_in_sparse_format(sp=sp,
        sentences=chunk_unencoded_lines)

      # run the session
      chunk_line_embeddings = session.run(
        embeddings,
        feed_dict={
          input_placeholder.values: values,
          input_placeholder.indices: indices,
          input_placeholder.dense_shape: dense_shape
        }
      )

      # output logs if verbose and hash the object into the full output dataframe
      for i, line_embedding in enumerate(np.array(chunk_line_embeddings).tolist()):
        if args.verbose:
          tf.logging.log(tf.logging.INFO, "Line: {}".format(chunk_unencoded_lines[i]))
          tf.logging.log(tf.logging.INFO, "Embedding size: {}".format(len(line_embedding)))
          line_embedding_snippet = ", ".join((str(x) for x in line_embedding[:3]))
          tf.logging.log(tf.logging.INFO, "Embedding: [{}, ...]\n".format(line_embedding_snippet))

        # Encode a hash for the string
        hash_object = hashlib.md5(chunk_unencoded_lines[i].encode('utf-8'))
        # Add a row to the dataframe
        output_dict[hash_object.hexdigest()] = {'line': chunk_unencoded_lines[i],
                                                'line_embedding': line_embedding}
  return output_dict

