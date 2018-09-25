source venv/bin/activate

# Number of lines from input source to use
export NUMLINES='32768'
export NUMTREES='50'
export NUMNEIGHBORS='10'
export SEARCHK='-1'

# Define the environmental variables
export PATHTXT="data/CMDC/all_lines_${NUMLINES}.txt"
export TFHUB_CACHE_DIR=data/module

# Extract the raw lines to a single file:
python process_cornell_data.py data/CMDC/movie_lines.txt ${PATHTXT} ${NUMLINES}

# Embed the lines using the encoder (Universal Sentence Encoder)
python embed_lines.py ${PATHTXT} &&

# Process the embeddings and save as unique strings and numpy array
python process_embeddings.py --path_to_text=${PATHTXT} --verbose &&

# Index the embeddings using an approximate nearest neighbor (annoy)
python index_embeddings.py --path_to_text=${PATHTXT} --verbose --num_trees=${NUMTREES} &&

# Build a simple command line interaction for model testing
python interact_with_model.py --path_to_text=${PATHTXT} --verbose --num_neighbors=${NUMNEIGHBORS}