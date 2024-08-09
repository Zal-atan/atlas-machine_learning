# This is a README for the QA Bot repo.

### In this repo we will practicing making a basic Question and Answer Bot using Transformers in Machine Learning
<br>

### Author - Ethan Zalta
<br>


# Tasks
### There are 5 tasks in this project

## Task 0
* Write a function def question_answer(question, reference): that finds a snippet of text within a reference document to answer a question:

    * question is a string containing the question to answer
    * reference is a string containing the reference document from which to find the answer
    * Returns: a string containing the answer
    * If no answer is found, return None
    * Your function should use the bert-uncased-tf2-qa model from the tensorflow-hub library
    * Your function should use the pre-trained BertTokenizer, bert-large-uncased-whole-word-masking-finetuned-squad, from the transformers library

## Task 1
* Create a script that takes in input from the user with the prompt Q: and prints A: as a response. If the user inputs exit, quit, goodbye, or bye, case insensitive, print A: Goodbye and exit.

## Task 2
* Based on the previous tasks, write a function def answer_loop(reference): that answers questions from a reference text:

    * reference is the reference text
    * If the answer cannot be found in the reference text, respond with Sorry, I do not understand your question.

## Task 3
* Write a function def semantic_search(corpus_path, sentence): that performs semantic search on a corpus of documents:

    * corpus_path is the path to the corpus of reference documents on which to perform semantic search
    * sentence is the sentence from which to perform semantic search
    * Returns: the reference text of the document most similar to sentence

## Task 4
* Based on the previous tasks, write a function def question_answer(coprus_path): that answers questions from multiple reference texts:

    * corpus_path is the path to the corpus of reference documents
