import os

from Jann.embed_lines import embed_lines
from Jann.index_embeddings import index_embeddings
from Jann.interact_with_model import interact_with_model
from Jann.process_cornell_data import process_cornell_data
from Jann.process_embeddings import process_embeddings
from Jann.process_pairs_data import process_pairs_data

from collections import namedtuple


FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_files',
)

test_lines = os.path.join(FIXTURE_DIR, 'test_lines.txt')
test_pairs = os.path.join(FIXTURE_DIR, 'test_pairs.txt')
test_CMDC_movie_lines = os.path.join(FIXTURE_DIR, 'movie_lines.txt')
test_CMDC_movie_lines_out = os.path.join(
    FIXTURE_DIR, 'all_lines.txt')
test_CMDC_movie_pairs_out = os.path.join(
    FIXTURE_DIR, 'all_pairs.txt')
movie_lines_fields = [
    "lineID", "characterID", "movieID", "character", "text"]
test_CMDC_movie_conversations = os.path.join(
    FIXTURE_DIR, 'test_CMDC_movie_conversations.txt')
movie_conversations_fields = [
    "character1ID", "character2ID", "movieID", "utteranceIDs"]

NUMLINES = 0
NUMTREES = 100
NUMNEIGHBORS = 10
SEARCHK = -1

# Set some default TestArgs
filed_names = ('infile_path outfile num_lines pairs ' +
               'module_path use_sentence_piece infile ' +
               'delimiter verbose num_trees num_neighbors ' +
               'search_k')
filed_names = filed_names.split(' ')
TestArgs = namedtuple(typename='TestArgs',
                      field_names=filed_names,
                      defaults=(None,) * len(filed_names))


def test_process_cornell_data():
    """Test the processing of Cornell Movie Dialog Data."""
    args = TestArgs(infile_path=FIXTURE_DIR,
                    outfile=test_CMDC_movie_lines_out,
                    num_lines=NUMLINES,
                    pairs=False)
    status = process_cornell_data(args)
    assert status


def test_process_pairs_data():
    """Test the processing of Cornell Movie Dialog Pairs Data."""
    args = TestArgs(infile=test_pairs,
                    outfile=test_CMDC_movie_pairs_out,
                    num_lines=NUMLINES,
                    pairs=True,
                    delimiter='\t')
    status = process_pairs_data(args)
    assert status


def test_embed_lines():
    """Test the embedding of lines."""
    module_name = 'universal-sentence-encoder-lite-2'
    args = TestArgs(infile=test_pairs,
                    outfile=test_CMDC_movie_pairs_out,
                    num_lines=NUMLINES,
                    pairs=True,
                    delimiter='\t',
                    module_path=os.path.join(FIXTURE_DIR, module_name),
                    use_sentence_piece=True)
    status = embed_lines(args)
    assert status


def test_process_embeddings():
    """Test the processing of the embedding of lines."""
    module_name = 'universal-sentence-encoder-lite-2'
    args = TestArgs(infile=test_pairs,
                    outfile=test_CMDC_movie_pairs_out,
                    num_lines=NUMLINES,
                    pairs=True,
                    delimiter='\t',
                    module_path=os.path.join(FIXTURE_DIR, module_name),
                    use_sentence_piece=True)
    status = process_embeddings(args)
    assert status


def test_index_embeddings():
    """Test the indexing of the embeddings of lines."""
    module_name = 'universal-sentence-encoder-lite-2'
    args = TestArgs(infile=test_pairs,
                    outfile=test_CMDC_movie_pairs_out,
                    num_lines=NUMLINES,
                    pairs=True,
                    delimiter='\t',
                    module_path=os.path.join(FIXTURE_DIR, module_name),
                    use_sentence_piece=True,
                    num_trees=NUMTREES)
    status = index_embeddings(args)
    assert status


def test_interact_with_model():
    """Test model interaction."""
    module_name = 'universal-sentence-encoder-lite-2'
    args = TestArgs(infile=test_pairs,
                    outfile=test_CMDC_movie_pairs_out,
                    num_lines=NUMLINES,
                    pairs=True,
                    delimiter='\t',
                    module_path=os.path.join(FIXTURE_DIR, module_name),
                    use_sentence_piece=True,
                    num_trees=NUMTREES)
    status = interact_with_model(args, debug=True)
    assert status
