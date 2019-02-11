[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fkorymath%2Fjann.svg?type=shield)](https://app.fossa.io/projects/git%2Bgithub.com%2Fkorymath%2Fjann?ref=badge_shield) [![CircleCI](https://circleci.com/gh/korymath/jann.svg?style=svg)](https://circleci.com/gh/korymath/jann)

# jann
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fkorymath%2Fjann.svg?type=shield)](https://app.fossa.io/projects/git%2Bgithub.com%2Fkorymath%2Fjann?ref=badge_shield)

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

## The Wiki

There is more information on the [Wiki](https://github.com/korymath/jann/wiki).

## Start Contributing
The guide for contributors can be found [here](https://github.com/korymath/jann/blob/master/CONTRIBUTING.md). It covers everything you need to know to start contributing to `jann`.

## References

* [Universal Sentence Encoder on TensorFlow Hub](https://tfhub.dev/google/universal-sentence-encoder-lite/2)
* [Cer, Daniel, et al. 'Universal sentence encoder.' arXiv preprint arXiv:1803.11175 (2018).](https://arxiv.org/abs/1803.11175)
* [Danescu-Niculescu-Mizil, Cristian, and Lillian Lee. 'Chameleons in imagined conversations: A new approach to understanding coordination of linguistic style in dialogs.' Proceedings of the 2nd Workshop on Cognitive Modeling and Computational Linguistics. Association for Computational Linguistics, 2011.](https://dl.acm.org/citation.cfm?id=2021105)

## Credits

`jann` is made with love by [Kory Mathewson](https://korymathewson.com).

Icon made by [Freepik](http://www.freepik.com) from [www.flaticon.com](https://www.flaticon.com/) is licensed by [CC 3.0 BY](http://creativecommons.org/licenses/by/3.0/).

## License
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fkorymath%2Fjann.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2Fkorymath%2Fjann?ref=badge_large)