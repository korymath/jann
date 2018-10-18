import sys
import argparse
import numpy as np


def main(arguments):

  parser = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--infile', help="Input file")
  parser.add_argument('--outfile', help="Verbose")
  parser.add_argument('--num_lines', type=int, help="Verbose")
  parser.add_argument('--store_pairs', dest='store_pairs',
                      help="Store pairs", action='store_true')
  args = parser.parse_args(arguments)

  lines = []
  try:
    with open(args.infile, errors='ignore') as f:
      for line in f:
        values = line.split(" +++$+++ ")
        lines.append(values[-1].strip())
  except FileNotFoundError as error:
    print(error)
    print('Input file not found, correct the specified location.')
    sys.exit(0)

  print("Found {} input lines.".format(len(lines)))

  with open(args.outfile, 'w') as f:
    if args.num_lines != 0:
      for item in np.random.choice(lines, args.num_lines, replace=False):
        f.write("%s\n" % item)
    else:
      for item in lines:
        f.write("%s\n" % item)

  print("Wrote {} lines to {}.".format(len(lines), args.outfile))


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))