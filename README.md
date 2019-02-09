# jann
Hi. I am `jann`. I am a retrieval-based chatbot. I would make a great baseline.

## Allow me to (re)introduce myself

I uses approximate nearest neighbor lookup using [Spotify's Annoy (Apache License 2.0)](https://github.com/spotify/annoy) library, over a distributed semantic embedding space ([Google's Universal Sentence Encoder (code: Apache License 2.0)](https://alpha.tfhub.dev/google/universal-sentence-encoder/2) from [TensorFlow Hub](https://www.tensorflow.org/hub/).

## Objectives

The goal of `jann` is to explicitly describes each step of the process of building a semantic similarity retrieval-based text chatbot. It is designed to be able to use diverse text source as input (e.g. Facebook messages, tweets, emails, movie lines, speeches, restaurant reviews, ...) so long as the data is collected in a single text file to be ready for processing.

## Install and configure requirements

`jann` is tested on macOS 10.14.

```sh
# Execute installation script
chmod +x install.sh
./install.sh
```

## Run jann

```sh
chmod +x run.sh
./run.sh
```

## Interaction

For interaction with the model, the only files needed are the unique strings (`_unique_strings.csv`) and the Annoy index (`.ann`) file. With the unique strings and the index file you can build a basic interaction. This is demonstrated in the `interact_with_model.py` file.

## Run Web Server

```sh
uwsgi --socket :8001 -w wsgi:JANN
```

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

As an example, we might use the first 50 lines of movie dialogue from the [Cornell Movie Dialog Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html). You can set the number of lines from the corpus you want to use by changing the parameter `export NUMLINES='2048'` in `run_CMDC.sh`.

```sh
chmod +x run_CMDC.sh
./run.sh
```

## Pairs

Conversational dialogue is composed of sequences of utterances. The sequence can be seen as pairs of utterances: inputs and responses. Nearest neighbors to a given input will find neighbors which are semantically related to the input. By storing input<>response pairs, rather than only inputs, `jann` can respond with a response to similar inputs. This example is shown in `run_CMDC_pairs.sh`.

## Custom Datasets

You can use any dataset you want! Format your source text with a single entry on each line, as follows:

```sh
# data in YOUR_FAVORITE_FILENAME.txt
This is the first line.
This is a response to the first line.
This is a response to the second line.
```

Change change the line `export INFILE="data/CMDC/YOUR_FAVORITE_FILENAME.txt"` in `run.sh`.

You might connect it with a source from [Botnik Studio's Sources](http://github.com/botnikstudios/sources). You can find an example of the entire `jann` pipeline using the `pairs` configuration on a custom datasource in `run_byron_pairs.sh`.

## Issues

* Add sources
* uwsgi --socket :8001 -w wsgi:JANN
zsh: command not found: uwsgi

### Error/Warning:
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

# Notes:

## Prepare the Universal Sentence Encoder embedding module
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

## References

* [Cer, Daniel, et al. 'Universal sentence encoder.' arXiv preprint arXiv:1803.11175 (2018).](https://arxiv.org/abs/1803.11175)
* [Danescu-Niculescu-Mizil, Cristian, and Lillian Lee. 'Chameleons in imagined conversations: A new approach to understanding coordination of linguistic style in dialogs.' Proceedings of the 2nd Workshop on Cognitive Modeling and Computational Linguistics. Association for Computational Linguistics, 2011.](https://dl.acm.org/citation.cfm?id=2021105)

## Credits

`jann` is made with love by [Kory Mathewson](https://korymathewson.com).

Icon made by [Freepik](http://www.freepik.com) from [www.flaticon.com](https://www.flaticon.com/) is licensed by [CC 3.0 BY](http://creativecommons.org/licenses/by/3.0/).