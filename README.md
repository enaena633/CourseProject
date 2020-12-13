# CS410 Course Project

Team JEM: Reproducing Paper on Casual Topic Mining 

Zhangzhou Yu (leader)
Matthew McCarty
Jack Ma

## Presentation/Demo

A presentation and demonstration of installing and running the application is available at https://mediaspace.illinois.edu/media/t/1_yra0qvjp .

## Overview
This repository contains code to replicate an experiment done in a paper regarding causal topic mining with time series feedback:

Hyun Duk Kim, Malu Castellanos, Meichun Hsu, ChengXiang Zhai, Thomas Rietz, and Daniel Diermeier. 2013. 
Mining causal topics in text data: Iterative topic modeling with time series feedback. 
In _Proceedings of the 22nd ACM international conference on information & knowledge management_ (CIKM 2013). 
ACM, New York, NY, USA, 885-890. DOI=10.1145/2505515.2505612

The intent of this paper is to develop a method to consider additional contextual data (specifically, in the form of a time series) to supplement topic mining. The paper discusses two scenarios (presidential elections and stock prices); we chose to replicate the former.

The specific experiment that was replicated involves determining topics from NYT articles from May-October 2000, with the additional context of betting odds for Bush and Gore winning the 2000 Presidential election. There are two files which are used as input for the Python code. One is the time series data for the betting odds, which is located in `Iowa2000PresidentOdds.csv`. The second input file, `consolidated_nyt.tsv`, is a list of NYT articles between May and October 2000. The NYT articles were filtered by 'Bush' and 'Gore' keywords to ensure that non-relevant documents were not considered for topic generation. The article date is also included with the article content, so that the time context of the article's publication can be considered with the presidential odds time series.

The output of the program will be a list of topics, and the top three words within each topics. Unlike the simple PSLA algorithm, these topics highlight words that are highly correlated with the change of betting odds for Bush or Gore winning the election. The number of topics is determined by a parameter _tn_, and the paper discusses the performance of the algorithm with varying values of _tn_. For the purposes of our experiment reproduction, we chose _tn=10_.

## Software Implementation
The experiment was reproduced in Python (version 3.8.6) with the help of several libraries, which are listed below:

* `numpy` - for general linear algebra operations
* `gensim` - for generating a mapping between a token id and the _normalized_ words which they represent
* `statsmodels` - for the time-series causality test

The algorithm itself is a modified version of the PLSA algorithm, which was initially implemented for a homework assignment (MP3) in CS410 at UIUC. The `plsawithprior.py` file contains a `plsa` class which contains many variables of use, some of which are highlighted below:

* `document_topic_prob` - the probability of `p(z | d)` where `z` represents a specific topic and `d` represents a specific document
* `mu` - the strength of the prior probability (when `mu=0`, the result would match PLSA with no prior)
* `prior` - the _prior_ probability of `p(w | z)` where `w` represents a word and `z` represents a specific topic
* `topic_word_prob` = the _posterior_ probability of `p(w | z)`

Future modifications could be made to the algorithm to change how the prior is generated (based on other time-series/non-document data source).

## Software Usage
1. Run `git clone https://github.com/enaena633/CourseProject.git` to clone the code repository.
2. Install [Python 3](https://www.python.org/downloads/release/python-386/). 
3. Install the following python libraries (via `pip`, etc.):

* `numpy`
* `gensim`
* `statsmodels`

4. Run `python main.py` in the repository directory.

## Results

The following list is the top 3 words in the ten topics that were mined from the New York Times documents:

```
ad win ralli
night lehrer trillion
econom recent try
support governor alaska
state governor alaska
governor clarenc right
night win tuesdai
wetston abm recent
offic men try
win ralli church
```

These results are different from the paper's results, which are included below:

```
tax cut 1
screen pataki giuliani
enthusiasm door symbolic
oil energy prices
pres al vice
love tucker presented
partial abortion privatization
court supreme abortion
gun control nra
news w top
```

This can be explained by the following:

* The implementation of several elements of the algorithm (Granger causality test, PLSA, etc.) were implemented in Python, whereas the paper used R.
* We used the `gensim` package to perform stemming of words (which would cause words like `econom` to appear instead of `economy` or `economic`).
* `gensim` was also used to remove stop words. The paper does not specify whether a background language model was used in its implementation of PLSA or if any stop word removal was done.
* The EM algorithm is guaranteed to converge to a local (but not necessarily global) maximum, which causes output to be different even with the same implementation when different random starting values are used.
* Certain parameters in the paper are not specified (e.g., the threshold value gamma for the significance cutoff for words at the topic level, we used 90%).

## Team Member Contributions
All team members were engaged and involved in reproducing the experiment from the paper sourced above. In addition to weekly meetings which everyone participated in, individual team members were responsible for the following:

Zhangzhou Yu (leader) - Granger causality test, administrative/organizational tasks

Matthew McCarty - Data retrieval/cleaning, library research, documentation

Jack Ma - PLSA augmentation to include use of contextual time series data