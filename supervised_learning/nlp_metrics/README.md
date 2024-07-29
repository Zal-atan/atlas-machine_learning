# This is a README for the NLP Metrics repo.

### In this repo we will practicing basic uses of Natural Language Processing in Machine Learning
<br>

### Author - Ethan Zalta
<br>


# Tasks
### There are 3 tasks in this project

## Task 0
* Write the function def uni_bleu(references, sentence): that calculates the unigram BLEU score for a sentence:

    * references is a list of reference translations
        * each reference translation is a list of the words in the translation
    * sentence is a list containing the model proposed sentence
    * Returns: the unigram BLEU score

## Task 1
* Write the function def ngram_bleu(references, sentence, n): that calculates the n-gram BLEU score for a sentence:

    * references is a list of reference translations
        * each reference translation is a list of the words in the translation
    * sentence is a list containing the model proposed sentence
    * n is the size of the n-gram to use for evaluation
    * Returns: the n-gram BLEU score

## Task 2
* Write the function def cumulative_bleu(references, sentence, n): that calculates the cumulative n-gram BLEU score for a sentence:

    * references is a list of reference translations
        * each reference translation is a list of the words in the translation
    * sentence is a list containing the model proposed sentence
    * n is the size of the largest n-gram to use for evaluation
    * All n-gram scores should be weighted evenly
    * Returns: the cumulative n-gram BLEU score

