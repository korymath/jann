import os
from Jann.utils import load_data

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_files',
    )


def test_load_data_list():
    test_file = os.path.join(FIXTURE_DIR, 'test_lines.txt')
    lines, response_lines = load_data(
        test_file, 'list')
    assert len(lines) == 50


# def test_load_lines(self):
#     pass

# def test_load_conversations(self):
#     pass

# def test_extract_pairs(self):
#     pass

# def test_extract_pairs_from_lines(self):
#     pass

# def test_process_to_IDs_in_sparse_format(self):
#     pass

# def test_get_id_chunks(self):
#     pass

# def test_embed_lines(self):
#     pass

# def test_generative_model_use(self):
#     pass
