import sys


def main(arguments):
  fname = sys.argv[1]
  output_fname = sys.argv[2]
  num_lines = int(sys.argv[3])

  print(fname, output_fname, num_lines)

  lines = []

  try:
    with open(fname, errors='ignore') as f:
      for line in f:
        values = line.split(" +++$+++ ")
        lines.append(values[-1].strip())
  except FileNotFoundError as error:
    print(error)
    print('Input file not found, correct the specified location.')
    sys.exit(0)

  print("Found {} input lines.".format(len(lines)))

  with open(output_fname, 'w') as f:
    if num_lines != 0:
      for item in lines[:num_lines]:
        f.write("%s\n" % item)
    else:
      for item in lines:
        f.write("%s\n" % item)

  print("Wrote {} lines to {}.".format(len(lines), output_fname))


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))