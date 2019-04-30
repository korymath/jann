# source venv/bin/activate

# Number of lines from input source to use
export NUMLINES='1000'
export NUMTREES='100'
export NUMNEIGHBORS='10'
export SEARCHK='-1'

# Define the environmental variables
export INFILEPATH="data/CMDC"
export INFILE="data/CMDC/all_lines_${NUMLINES}.txt"
export TFHUB_CACHE_DIR=data/module

# Extract the raw lines to a single file:
python process_cornell_data.py --infile_path=${INFILEPATH} \
--outfile=${INFILE} --num_lines=${NUMLINES} &&

# Embed the lines using the encoder (Universal Sentence Encoder)
python embed_lines.py --infile=${INFILE} --verbose &&

# Process the embeddings and save as unique strings and numpy array
python process_embeddings.py --infile=${INFILE} --verbose &&

# Index the embeddings using an approximate nearest neighbor (annoy)
python index_embeddings.py --infile=${INFILE} --verbose \
--num_trees=${NUMTREES} &&

# Build a simple command line interaction for model testing
python interact_with_model.py --infile=${INFILE} --verbose \
--num_neighbors=${NUMNEIGHBORS}
