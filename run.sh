source venv/bin/activate

# Number of lines from input source to use
export NUMTREES='50'
export NUMNEIGHBORS='10'
export SEARCHK='-1'

# Define the environmental variables
export PATHTXT="data/CMDC/all_lines_50.txt"
export PATHTXT="data/CMDC/testing_new_lines.txt"
export TFHUB_CACHE_DIR="data/module"

# Embed the lines using the encoder (Universal Sentence Encoder)
python embed_lines.py ${PATHTXT} --verbose &&

# Process the embeddings and save as unique strings and numpy array
python process_embeddings.py --path_to_text=${PATHTXT} --verbose &&

# Index the embeddings using an approximate nearest neighbor (annoy)
python index_embeddings.py --path_to_text=${PATHTXT} --verbose --num_trees=${NUMTREES} &&

# Build a simple command line interaction for model testing
python interact_with_model.py --path_to_text=${PATHTXT} --verbose --num_neighbors=${NUMNEIGHBORS}