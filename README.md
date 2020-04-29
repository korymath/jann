# jann
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

Note: `jann` development is tested on macOS 10.15.4 Catalina. Deployment is tested on Ubuntu.

To run `jann` on your local system or a server, you will need to perform the following installation steps.

```sh
# OSX: Install homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"

# OSX: Install wget
brew install wget

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
mkdir -p ${TFHUB_CACHE_DIR}

# Download and unpack the Universal Sentence Encoder Lite model (~25 MB)
if [ ! -f ${TFHUB_CACHE_DIR}/module_lite.tar.gz ]; then
  echo "No module found, downloading..."
  wget 'https://tfhub.dev/google/universal-sentence-encoder-lite/2?tf-hub-format=compressed' -O ${TFHUB_CACHE_DIR}/module_lite.tar.gz
  cd ${TFHUB_CACHE_DIR}
  mkdir -p universal-sentence-encoder-lite-2 && tar -zxvf module_lite.tar.gz -C universal-sentence-encoder-lite-2
  cd -
else
  echo "Module found!"
fi
```

### Download Cornell Movie Dialog Database

Download the [Cornell Movie Dialog Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html), and extract to `data/CMDC`.

```sh
# Change directory to CMDC data subdirectory
mkdir data/CMDC
cd data/CMDC/

# Download the corpus
wget http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip

# Unzip the corpus and move to the main directory
unzip cornell_movie_dialogs_corpus.zip
mv cornell\ movie-dialogs\ corpus/movie_lines.txt movie_lines.txt

# Change direcory to jann's main directory
cd ../..
```

As an example, we might use the first 50 lines of movie dialogue from the [Cornell Movie Dialog Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html).

You can set the number of lines from the corpus you want to use by changing the parameter `export NUMLINES='50'` in `run_examples/run_CMDC.sh`.

## (simple) Run Basic Example

```sh
cd Jann
# make sure that the run code is runnable
chmod +x run_examples/run_CMDC.sh
# run it
./run_examples/run_CMDC.sh
```

## (advanced) Running Model Building

`jann` is composed of several submodules, each of which can be run in sequence as follows:

```sh
# Ensure that the virtual environment is activated
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

`jann` is designed to run as a web service to be queried by a dialogue interface builder. For instance, `jann` is natively configured to be compatible with Dialogflow. The web service runs using the Flask micro-framework and uses the performance-oriented gunicorn application server to launch the application with 4 workers.

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

## Prepare the Universal Sentence Encoder embedding module

Note from [TensorFlow Hub](https://tfhub.dev/google/universal-sentence-encoder/2): The module performs best effort text input preprocessing, therefore it is not required to preprocess the data before applying the module.


```sh
mkdir data/modules
export TFHUB_CACHE_DIR=data/modules

# Lite model (25 MB)
wget 'https://tfhub.dev/google/universal-sentence-encoder-lite/2?tf-hub-format=compressed' -O ${TFHUB_CACHE_DIR}/module_lite.tar.gz
cd ${TFHUB_CACHE_DIR}
mkdir -p universal-sentence-encoder-lite-2 && tar -zxvf module_lite.tar.gz -C universal-sentence-encoder-lite-2

# Standard Model (914 MB)
wget 'https://alpha.tfhub.dev/google/universal-sentence-encoder/2?tf-hub-format=compressed' -O ${TFHUB_CACHE_DIR}/module_standard.tar.gz
cd ${TFHUB_CACHE_DIR}
mkdir -p universal-sentence-encoder-2 && tar -zxvf module_standard.tar.gz -C universal-sentence-encoder-2

# Large Model (746 MB)
wget 'https://alpha.tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed' -O ${TFHUB_CACHE_DIR}/module_large.tar.gz
cd ${TFHUB_CACHE_DIR}
tar -zxvf module_large.tar.gz
mkdir -p universal-sentence-encoder-large-3 && tar -zxvf module_large.tar.gz -C universal-sentence-encoder-large-3
```

## Annoy parameters

There are two parameters for the Approximate Nearest Neighbour:
* set `n_trees` as large as possible given the amount of memory you can afford,
* set `search_k` as large as possible given the time constraints you have for the queries. This parameter is a interaction tradeoff between accuracy and speed.

## Run details for cloud serving (e.g. Digital Ocean) using nginx and uwsgi

You will need to configure your server with the necessary software:

```sh
sudo apt update
sudo apt install python3-pip python3-dev python3-venv build-essential libssl-dev libffi-dev python3-setuptools
sudo apt-get install nginx
sudo /etc/init.d/nginx start    # start nginx
```

Then, you can reference a more in-depth guide [here](https://uwsgi-docs.readthedocs.io/en/latest/tutorials/Django_and_nginx.html)

You will need the uwsgi_params file, which is available in the nginx directory of the uWSGI distribution, or from https://github.com/nginx/nginx/blob/master/conf/uwsgi_params

Copy it into your project directory. In a moment we will tell nginx to refer to it.

`/etc/nginx/sites-available/JANN.conf`
```sh
# JANN.conf

# the upstream component nginx needs to connect to
upstream flask {
    # for a web port socket
    server 127.0.0.1:8001;
}

# configuration of the server
server {
    # the port your site will be served on
    listen      8000;
    # the domain name it will serve for
    server_name IP; # substitute your machine's IP address or FQDN
    charset     utf-8;

    # Finally, send all non-media requests to the Django server.
    location / {
        uwsgi_pass  flask;
        # the uwsgi_params file you installed
        include     /home/${USER}/jann/uwsgi_params;
    }
}
```

```
sudo ln -s ~/path/to/your/mysite/mysite_nginx.conf /etc/nginx/sites-enabled/
sudo /etc/init.d/nginx restart
uwsgi --socket :8001 -w wsgi:JANN
```


## Common Errors/Warnings and Solutions

```sh
/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/importlib/_bootstrap.py:205: RuntimeWarning: compiletime version 3.5 of module 'tensorflow.python.framework.fast_tensor_util' does not match runtime version 3.6
  return f(*args, **kwds)
```
Solution (for OSX 10.13):
```sh
pip install --ignore-installed --upgrade https://github.com/lakshayg/tensorflow-build/releases/download/tf1.9.0-macos-py27-py36/tensorflow-1.9.0-cp36-cp36m-macosx_10_13_x86_64.whl
```

### Error/Warning:
```sh
FileNotFoundError: [Errno 2] No such file or directory: 'data/CMDC/movie_lines.txt'
```
Solution:
```sh
Ensure that the input movie lines file is extracted to the correct path
```

### Error/Warning
```sh
ValueError: Signature 'spm_path' is missing from meta graph.
```

#### Solution:
Currently `jann` is configured to use the `universal-sentence-encoder-lite` module from TFHub as it is small, lightweight, and ready for rapid deployment. This module depends on the [SentencePiece](https://github.com/google/sentencepiece) library and the SentencePiece model published with the module.

You will need to make some minor code adjustments to use the heaviery modules (such as [universal-sentence-encoder](https://alpha.tfhub.dev/google/universal-sentence-encoder/2)
and [universal-sentence-encoder-large](https://alpha.tfhub.dev/google/universal-sentence-encoder-large/3).


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
