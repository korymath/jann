import os
from Jann.utils import load_data
from Jann.utils import load_lines
from Jann.utils import extract_pairs
from Jann.utils import parse_arguments
from Jann.utils import load_conversations
from Jann.utils import extract_pairs_from_lines


FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_files',
    )

test_lines = os.path.join(FIXTURE_DIR, 'test_lines.txt')
test_pairs = os.path.join(FIXTURE_DIR, 'test_pairs.txt')
test_CMDC_movie_lines = os.path.join(FIXTURE_DIR, 'test_CMDC_movie_lines.txt')
movie_lines_fields = [
    "lineID", "characterID", "movieID", "character", "text"]
test_CMDC_movie_conversations = os.path.join(
    FIXTURE_DIR, 'test_CMDC_movie_conversations.txt')
movie_conversations_fields = [
    "character1ID", "character2ID", "movieID", "utteranceIDs"]


def test_parse_arguments():
    arguments = ['--verbose']
    args = parse_arguments(arguments=arguments)
    # ensure that default values and arguments are set
    assert (args.search_k == 10) and (args.verbose)


def test_load_data_list_not_pairs():
    lines, response_lines = load_data(
        test_lines, 'list', pairs=False, delimiter='\t')
    assert (len(lines) == 50) and not response_lines


def test_load_data_list_pairs():
    lines, response_lines = load_data(
        test_pairs, 'list', pairs=True, delimiter='\t')
    assert (len(lines) == 2) and (len(response_lines) == 2)


def test_load_lines():
    lines = load_lines(test_CMDC_movie_lines, movie_lines_fields)
    assert len(lines) == 3


def test_load_conversations():
    lines = load_lines(test_CMDC_movie_lines, movie_lines_fields)
    convos = load_conversations(
        fname=test_CMDC_movie_conversations,
        lines=lines,
        fields=movie_conversations_fields)
    assert len(convos) == 2


def test_extract_pairs():
    lines = load_lines(test_CMDC_movie_lines, movie_lines_fields)
    convos = load_conversations(
        fname=test_CMDC_movie_conversations,
        lines=lines,
        fields=movie_conversations_fields)
    collected_pairs = extract_pairs(convos)
    assert len(collected_pairs) == 3


def test_extract_pairs_from_lines():
    lines, response_lines = load_data(
        test_lines, 'list', pairs=False, delimiter='\t')
    collected_pairs = extract_pairs_from_lines(lines)
    assert len(collected_pairs) == 49