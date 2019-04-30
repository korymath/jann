# jann
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fkorymath%2Fjann.svg?type=shield)](https://app.fossa.io/projects/git%2Bgithub.com%2Fkorymath%2Fjann?ref=badge_shield)
[![CircleCI](https://circleci.com/gh/korymath/jann.svg?style=svg)](https://circleci.com/gh/korymath/jann)
[![codecov](https://codecov.io/gh/korymath/jann/branch/master/graph/badge.svg)](https://codecov.io/gh/korymath/jann)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

Hi. I am `jann`. I am a retrieval-based chatbot. I would make a great baseline.

## Allow me to (re)introduce myself

I uses approximate nearest neighbor lookup using [Spotify's Annoy (Apache License 2.0)](https://github.com/spotify/annoy) library, over a distributed semantic embedding space ([Google's Universal Sentence Encoder (code: Apache License 2.0)](https://alpha.tfhub.dev/google/universal-sentence-encoder/2) from [TensorFlow Hub](https://www.tensorflow.org/hub/).

## Objectives

The goal of `jann` is to explicitly describes each step of the process of building a semantic similarity retrieval-based text chatbot. It is designed to be able to use diverse text source as input (e.g. Facebook messages, tweets, emails, movie lines, speeches, restaurant reviews, ...) so long as the data is collected in a single text file to be ready for processing.

## Install and configure requirements

Note: `jann` is tested on macOS 10.14.

To run `jann` on your local system or a server, you will need to perform the following installation steps.

```sh
# Configure and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade Pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install Jann
python setup.py install

# Set environmental variable for TensorFlow Hub
export TFHUB_CACHE_DIR=Jann/data/module

# Make the TFHUB_CACHE_DIR
mkdir ${TFHUB_CACHE_DIR}

# Download and unpack the Universal Sentence Encoder Lite model (~25 MB)
if [ ! -f ${TFHUB_CACHE_DIR}/module_lite.tar.gz ]; then
    echo "No module found, downloading..."
    wget 'https://tfhub.dev/google/universal-sentence-encoder-lite/2?tf-hub-format=compressed' -O ${TFHUB_CACHE_DIR}/module_lite.tar.gz
    cd ${TFHUB_CACHE_DIR}
    mkdir -p universal-sentence-encoder-lite-2 && tar -zxvf module_lite.tar.gz -C universal-sentence-encoder-lite-2
    cd ../../..
fi
```

## (simple) Run Basic Example

```sh
cd Jann
# chmod +x run_examples/run_CMDC.sh
./run_examples/run_CMDC.sh
```


## (advanced) Running Model Building

`jann` is composed of several submodules, each of which can be run in sequence as follows:

```sh
source venv/bin/activate

# Change directory to Jann
cd Jann

# Number of lines from input source to use
export NUMTREES='100'
# Number of neighbors to return
export NUMNEIGHBORS='10'

# Define the environmental variables
export INFILE="data/CMDC/all_lines_50.txt"

# Embed the lines using the encoder (Universal Sentence Encoder)
python embed_lines.py --infile=${INFILE} --verbose &&

# Process the embeddings and save as unique strings and numpy array
python process_embeddings.py --infile=${INFILE} --verbose &&

# Index the embeddings using an approximate nearest neighbor (annoy)
python index_embeddings.py --infile=${INFILE} --verbose --num_trees=${NUMTREES} &&

# Build a simple command line interaction for model testing
python interact_with_model.py --infile=${INFILE} --verbose --num_neighbors=${NUMNEIGHBORS}
```

## Interaction

For interaction with the model, the only files needed are the unique strings (`_unique_strings.csv`) and the Annoy index (`.ann`) file. With the unique strings and the index file you can build a basic interaction. This is demonstrated in the `interact_with_model.py` file.

## Run Web Server

`jann` is desiged to run as a web service to be queried by a dialogue interface builder. For instance, `jann` is natively configured to be compatible with Dialogflow. The web service runs using the Flask micro-framework and uses a performant gunicorn application server to launch the application with 4 workers.

```sh
cd Jann
gunicorn --bind 0.0.0.0:8000 app:JANN -w 4
```

Once `jann` is running, in a new terminal window you can test the load on the server with [Locust](https://locust.io/), as defined in `Jann/tests/locustfile.py`:

```sh
source venv/bin/activate
cd Jann/tests
locust --host=http://0.0.0.0:8000
```

You can then navigate a web browser to [http://0.0.0.0:8089/](http://0.0.0.0:8089/), and simulate `N` users spawning at `M` users per second and making requests to `jann`.

### Cornell Movie Dialog Database

Download the [Cornell Movie Dialog Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html), and extract to `data/CMDC`.

```sh
# Change directory to CMDC data subdirectory
cd data/CMDC/

# Download the corpurs
wget http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip

# Unzip the corpus and move to the main directory
unzip cornell_movie_dialogs_corpus.zip
mv cornell\ movie-dialogs\ corpus/movie_lines.txt movie_lines.txt

# Change direcory to jann's main directory
cd ../..
```

As an example, we might use the first 50 lines of movie dialogue from the [Cornell Movie Dialog Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html). You can set the number of lines from the corpus you want to use by changing the parameter `export NUMLINES='2048'` in `run_examples/run_CMDC.sh`.

## Pairs

Conversational dialogue is composed of sequences of utterances. The sequence can be seen as pairs of utterances: inputs and responses. Nearest neighbours to a given input will find neighbours which are semantically related to the input. By storing input<>response pairs, rather than only inputs, `jann` can respond with a response to similar inputs. This example is shown in `run_examples/run_CMDC_pairs.sh`.

## Custom Datasets

You can use any dataset you want! Format your source text with a single entry on each line, as follows:

```sh
# data in YOUR_FAVORITE_FILENAME.txt
This is the first line.
This is a response to the first line.
This is a response to the second line.
```

Change change the line `export INFILE="data/CMDC/YOUR_FAVORITE_FILENAME.txt"` in `run.sh`.

You might connect it with a source from [Botnik Studio's Sources](http://github.com/botnikstudios/sources). You can find an example of the entire `jann` pipeline using the `pairs` configuration on a custom datasource in `run_examples/run_byron_pairs.sh`.

## The Wiki

There is more information on the [Wiki](https://github.com/korymath/jann/wiki).

## Start Contributing
The guide for contributors can be found [here](https://github.com/korymath/jann/blob/master/CONTRIBUTING.md). It covers everything you need to know to start contributing to `jann`.

## Tests

```sh
py.test --cov-report=xml --cov=Jann
```

## References

* [Universal Sentence Encoder on TensorFlow Hub](https://tfhub.dev/google/universal-sentence-encoder-lite/2)
* [Cer, Daniel, et al. 'Universal sentence encoder.' arXiv preprint arXiv:1803.11175 (2018).](https://arxiv.org/abs/1803.11175)
* [Danescu-Niculescu-Mizil, Cristian, and Lillian Lee. 'Chameleons in imagined conversations: A new approach to understanding coordination of linguistic style in dialogs.' Proceedings of the 2nd Workshop on Cognitive Modeling and Computational Linguistics. Association for Computational Linguistics, 2011.](https://dl.acm.org/citation.cfm?id=2021105)

## Credits

`jann` is made with love by [Kory Mathewson](https://korymathewson.com).

Icon made by [Freepik](http://www.freepik.com) from [www.flaticon.com](https://www.flaticon.com/) is licensed by [CC 3.0 BY](http://creativecommons.org/licenses/by/3.0/).

## License
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fkorymath%2Fjann.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2Fkorymath%2Fjann?ref=badge_large)