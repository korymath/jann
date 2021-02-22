import csv
import os
import sys

import numpy as np
import tensorflow.compat.v1 as tf  # type: ignore

import Jann.utils as utils

tf.disable_v2_behavior()


unique_lines = set({})

def filter_line(line):
    """Filter lines so that we have a unique set of English-language lines."""
    # Simple clean-up rules for trailing characters and punctuation.
    line = line.strip()
    line = line.rstrip()
    line = line.replace('\t', ' ')
    line = line.replace('  ', ' ')
    line = line.replace('  ', ' ')
    line = line.replace('”', '"')
    line = line.replace('„', '"')
    line = line.replace('…', '...')
    # Remove lines where more than half of characters are not ASCII.
    num_chars = len(line)
    num_chars_non_ascii = 0
    for c in line:
        try:
            c.encode(encoding='utf-8').decode('ascii')
        except UnicodeDecodeError:
            num_chars_non_ascii += 1
    if num_chars_non_ascii * 2 > num_chars:
        return None
    # Store only unique lines.
    if line not in unique_lines:
        unique_lines.add(line)
        return line
    else:
        return None


def process_novel_first_lines(args):
    """Main run function for processing the Novel First Lines dataset."""

    # Crowdsourced lines file.
    crowdsourced_lines_file = os.path.join(args.infile_path, 'crowdsourced_all.txt')
    tf.logging.info(
        'Novel First Lines crowdsourced_lines_path: {}'.format(crowdsourced_lines_file))

    lines = []
    try:
        with open(crowdsourced_lines_file, encoding='iso-8859-1') as f:
            for line in f:
                line = filter_line(line)
                if line is not None:
                    lines.append(line + '\t' + line)
    except FileNotFoundError as error:
        tf.logging.error(error)
        tf.logging.error(tf.logging.ERROR, 'File not found.')
        sys.exit(0)
    tf.logging.info("Found {} unique input lines.".format(len(lines)))
    lines = sorted(lines)

    with open(args.outfile, 'w', encoding='iso-8859-1') as f:
        for item in lines:
            f.write("%s\n" % item)
        tf.logging.info(
            'Wrote {} lines to {}.'.format(len(lines), args.outfile))
    return True


if __name__ == "__main__":
    # Parse the arguments
    args = utils.parse_arguments(sys.argv[1:])
    sys.exit(process_novel_first_lines(args))
