#!/usr/bin/env python3
""" Module creating the Q?A bot final verison """

quest_ans = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search

def question_answer(corpus_path):
    """
    Answers questions from multiple reference texts

    Inputs:\\
    corpus_path: the path to the corpus of reference documents
    """
    while True:
        question = input("Q: ")
        exits = ["exit", "quit", 'goodbye', 'bye']
        if question.lower() in exits:
            print("A: Goodbye")
            break
        reference = semantic_search(corpus_path, question)
        answer = quest_ans(question, reference)
        if answer == None or answer == "":
            answer = "Sorry, I do not understand your question."
        print(f"A: {answer}")
