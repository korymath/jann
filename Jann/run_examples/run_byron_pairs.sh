# source venv/bin/activate

# Number of lines from input source to use
export NUMPAIRS='182'
export NUMTREES='100'
export NUMNEIGHBORS='10'

# Define the environmental variables
export INFILE="data/botnik-sources/byron.txt"
export INFILE="data/botnik-sources/byron_${NUMPAIRS}_pairs.txt"

# Build the pairs from the input source
python process_pairs_data.py --infile=${INFILE} \
--outfile=${INFILE} --num_lines=${NUMPAIRS} --verbose &&

# Embed the lines using the encoder (Universal Sentence Encoder)
python embed_lines.py --infile=${INFILE} --verbose --pairs &&

# Process the embeddings and save as unique strings and numpy array
python process_embeddings.py --infile=${INFILE} --pairs --verbose &&

# Index the embeddings using an approximate nearest neighbor (annoy)
python index_embeddings.py --infile=${INFILE} --verbose \
--num_trees=${NUMTREES} &&

# Build a simple command line interaction for model testing
python interact_with_model.py --infile=${INFILE} --pairs --verbose \
--num_neighbors=${NUMNEIGHBORS}
