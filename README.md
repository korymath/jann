# jann

I am `jann`, a retreival-based chatbot baseline.

`jann` uses approximate nearest neighbor, using [Spotify's Annoy (Apache License 2.0)](https://github.com/spotify/annoy) library, over a distributed semantic embedding space as defined by [Google's Universal Sentence Encoder (code: Apache License 2.0)](https://alpha.tfhub.dev/google/universal-sentence-encoder/2) from TensorFlow Hub.

The goal of `jann` is to be a descriptive code sample which explicitly describes each step of the process of building a semantic similarity retrieval-based text chatbot. It is designed to be able to use any text as input (e.g. Facebook messages, tweets, emails, movie lines, speeches, restaurant reviews, ...) so long as it is collected in a single text file, ready for processing.

There are two parameters for the Approximate Nearest Neighbour:
* set `n_trees` as large as possible given the amount of memory you can afford,
* set `search_k` as large as possible given the time constraints you have for the queries.

## Install and configure requirements

```sh
chmod +x install.sh
./install.sh
```

## Collect and prepare data

Download the [Cornell Movie Dialog Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html), and extract to `data/CMDC`.

## Run jann

```sh
chmod +x run.sh
./run.sh
```

## Interaction

For interaction with the model, the only files needed are the unique strings (`_unique_strings.csv`) and the Annoy index (`.ann`) file. With the unique strings and the index file you can build a basic interaction. This is demonstrated in the `interact_with_model.py` file.

## References

* [Cer, Daniel, et al. "Universal sentence encoder." arXiv preprint arXiv:1803.11175 (2018).](https://arxiv.org/abs/1803.11175)
* [Danescu-Niculescu-Mizil, Cristian, and Lillian Lee. "Chameleons in imagined conversations: A new approach to understanding coordination of linguistic style in dialogs." Proceedings of the 2nd Workshop on Cognitive Modeling and Computational Linguistics. Association for Computational Linguistics, 2011.](https://dl.acm.org/citation.cfm?id=2021105)

## Issues

Error/Warning
```sh
/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/importlib/_bootstrap.py:205: RuntimeWarning: compiletime version 3.5 of module 'tensorflow.python.framework.fast_tensor_util' does not match runtime version 3.6
  return f(*args, **kwds)
```

Solution (for OSX 10.13):
```sh
pip install --ignore-installed --upgrade https://github.com/lakshayg/tensorflow-build/releases/download/tf1.9.0-macos-py27-py36/tensorflow-1.9.0-cp36-cp36m-macosx_10_13_x86_64.whl
```

Error/Warning:
```sh
FileNotFoundError: [Errno 2] No such file or directory: 'data/CMDC/movie_lines.txt'
```

Solution:
```sh
Ensure that the input movie lines file is extracted to the correct path
```

## Credits

`jann` is made with love by [Kory Mathewson](https://korymathewson.com).


# Notes:

## Prepare the Universal Sentence Encoder embedding module
```sh
mkdir data/module
export TFHUB_CACHE_DIR=data/module

# Lite model (25 MB)
wget 'https://tfhub.dev/google/universal-sentence-encoder-lite/2?tf-hub-format=compressed' -O ${TFHUB_CACHE_DIR}/module_lite.tar.gz

cd ${TFHUB_CACHE_DIR}
tar -zxvf module.tar.gz
```