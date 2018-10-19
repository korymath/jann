source venv/bin/activate

# Number of lines from input source to use
export NUMPAIRS='100'
export NUMTREES='100'
export NUMNEIGHBORS='10'

# Define the environmental variables
export INFILE="data/CMDC/botnik-sources/byron.txt"
export PATHTXT="data/botnik-sources/byron_${NUMPAIRS}_pairs.txt"

# Build the pairs from the input source
python process_pairs_data.py --infile_path=${INFILE} --outfile=${PATHTXT} --num_lines=${NUMPAIRS} --pairs --verbose &&

# Embed the lines using the encoder (Universal Sentence Encoder)
python embed_lines.py ${PATHTXT} --verbose &&

# Process the embeddings and save as unique strings and numpy array
python process_embeddings.py --path_to_text=${PATHTXT} --pairs --verbose &&

# Index the embeddings using an approximate nearest neighbor (annoy)
python index_embeddings.py --path_to_text=${PATHTXT} --verbose --num_trees=${NUMTREES} &&

# Build a simple command line interaction for model testing
python interact_with_model.py --path_to_text=${PATHTXT} --pairs --verbose --num_neighbors=${NUMNEIGHBORS}