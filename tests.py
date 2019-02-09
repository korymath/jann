import unittest

import utils

args = utils.parse_arguments()

class UtilsTest(unittest.TestCase):
  def test_load_data_list(self):
    lines, response_lines = utils.load_data(args.infile, 'list')
    self.assertEqual(len(lines), 50)

  def test_load_lines(self):
    pass

  def test_load_conversations(self):
    pass

  def test_extract_pairs(self):
    pass

  def test_extract_pairs_from_lines(self):
    pass

  def test_process_to_IDs_in_sparse_format(self):
    pass

  def test_get_id_chunks(self):
    pass

  def test_embed_lines(self):
    pass

  def test_generative_model_use(self):
    pass

if __name__ == '__main__':
  unittest.main()