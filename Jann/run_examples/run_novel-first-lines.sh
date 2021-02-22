#!/bin/bash
# Number of lines from input source to use
export NUMLINES='10000'
export NUMTREES='100'
export NUMNEIGHBORS='10'
export SEARCHK='-1'

# Define the environmental variables
export INFILEPATH="data/novel-first-lines-dataset"
export INFILE="data/novel-first-lines-dataset/all_lines_${NUMLINES}.txt"
export TFHUB_CACHE_DIR=data/module
export DATAURL="https://raw.githubusercontent.com/janelleshane/novel-first-lines-dataset/master/crowdsourced_all.txt"
export DATAFILE="crowdsourced_all.txt"

# Download the dataset
if [ ! -d ${INFILEPATH} ]; then
	echo "Creating directory $INFILEPATH..."
	mkdir $INFILEPATH
else
	echo "Directory $INFILEPATH already created, skipping."
fi
if [ ! -f ${INFILEPATH}/${DATAFILE} ]; then
	echo "Downloading novel-first-lines-dataset as $DATAFILE..."
	curl $DATAURL > $INFILEPATH/$DATAFILE
else
	echo "$DATAFILE already present, skipping."
fi


# Extract the raw lines to a single file:
python3 process_novel_first_lines.py --infile_path=${INFILEPATH} \
--outfile=${INFILE} --verbose

# Embed the lines using the encoder (Universal Sentence Encoder)
python3 embed_lines.py --infile=${INFILE} --verbose

# Process the embeddings and save as unique strings and numpy array
python3 process_embeddings.py --infile=${INFILE} --verbose

# Index the embeddings using an approximate nearest neighbor (annoy)
python3 index_embeddings.py --infile=${INFILE} --verbose \
--num_trees=${NUMTREES}

# Build a simple command line interaction for model testing
python3 interact_with_model.py --infile=${INFILE} --verbose \
--num_neighbors=${NUMNEIGHBORS}
