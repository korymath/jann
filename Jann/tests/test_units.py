import os

import Jann.utils as utils
from Jann.embed_lines import embed_lines
from Jann.index_embeddings import index_embeddings
from Jann.interact_with_model import interact_with_model
from Jann.process_cornell_data import process_cornell_data
from Jann.process_embeddings import process_embeddings
from Jann.process_pairs_data import process_pairs_data


FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_files',
    )

test_lines = os.path.join(FIXTURE_DIR, 'test_lines.txt')
test_pairs = os.path.join(FIXTURE_DIR, 'test_pairs.txt')
test_CMDC_movie_lines = os.path.join(FIXTURE_DIR, 'test_CMDC_movie_lines.txt')
test_CMDC_movie_lines_out = os.path.join(
  FIXTURE_DIR, 'all_lines_test_CMDC_movie_lines.txt')
movie_lines_fields = [
    "lineID", "characterID", "movieID", "character", "text"]
test_CMDC_movie_conversations = os.path.join(
    FIXTURE_DIR, 'test_CMDC_movie_conversations.txt')
movie_conversations_fields = [
    "character1ID", "character2ID", "movieID", "utteranceIDs"]

NUMLINES = '50'
NUMTREES = '100'
NUMNEIGHBORS = '10'
SEARCHK = '-1'

# Parse the arguments
args = utils.parse_arguments()


def test_process_cornell_data():
    args.infile_path = test_CMDC_movie_lines
    args.outfile = test_CMDC_movie_lines_out
    sstatus = process_cornell_data(args)
    assert sstatus


def test_process_pairs_data():
    assert True


def test_embed_lines():
    assert True


def test_process_embeddings():
    assert True


def test_index_embeddings():
    assert True


def test_interact_with_model():
    assert True
