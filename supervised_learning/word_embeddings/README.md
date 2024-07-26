# This is a README for the Word Embeddings repo.

### In this repo we will practicing basic uses of Word embeddings for natural language processing in Machine Learning
<br>

### Author - Ethan Zalta
<br>


# Tasks
### There are 6 tasks in this project

## Task 0
* Write a function def bag_of_words(sentences, vocab=None): that creates a bag of words embedding matrix:

    * sentences is a list of sentences to analyze
    * vocab is a list of the vocabulary words to use for the analysis
        * If None, all words within sentences should be used
    * Returns: embeddings, features

        * embeddings is a numpy.ndarray of shape (s, f) containing the embeddings
            * s is the number of sentences in sentences
            * f is the number of features analyzed
        * features is a list of the features used for embeddings
    * You are not allowed to use genism library.

## Task 1
* Write a function def tf_idf(sentences, vocab=None): that creates a TF-IDF embedding:

    * sentences is a list of sentences to analyze
    * vocab is a list of the vocabulary words to use for the analysis
        * If None, all words within sentences should be used
    * Returns: embeddings, features
        * embeddings is a numpy.ndarray of shape (s, f) containing the embeddings
            * s is the number of sentences in sentences
            * f is the number of features analyzed
        * features is a list of the features used for embeddings

## Task 2
* Write a function def tf_idf(sentences, vocab=None): that creates a TF-IDF embedding:

    * sentences is a list of sentences to analyze
    * vocab is a list of the vocabulary words to use for the analysis
    * If None, all words within sentences should be used
    * Returns: embeddings, features
    * embeddings is a numpy.ndarray of shape (s, f) containing the embeddings
    * s is the number of sentences in sentences
    * f is the number of features analyzed
    * features is a list of the features used for embeddings

## Task 3
* Write a function def gensim_to_keras(model): that converts a gensim word2vec model to a keras Embedding layer:

    * model is a trained gensim word2vec models
    * Returns: the trainable keras Embedding

## Task 4
* Write a function def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5, cbow=True, iterations=5, seed=0, workers=1): that creates and trains a genism fastText model:

    * sentences is a list of sentences to be trained on
    * size is the dimensionality of the embedding layer
    * min_count is the minimum number of occurrences of a word for use in training
    * window is the maximum distance between the current and predicted word within a sentence
    * negative is the size of negative sampling
    * cbow is a boolean to determine the training type; True is for CBOW; False is for Skip-gram
    * iterations is the number of iterations to train over
    * seed is the seed for the random number generator
    * workers is the number of worker threads to train the model
    * Returns: the trained model

## Task 5
* When training an ELMo embedding model, you are training:

1. The internal weights of the BiLSTM
2. The character embedding layer
3. The weights applied to the hidden states

    * In the text file 5-elmo, write the letter answer, followed by a newline, that lists the correct statements:

A. 1, 2, 3
B. 1, 2
C. 2, 3
D. 1, 3
E. 1
F. 2
G. 3
H. None of the above

