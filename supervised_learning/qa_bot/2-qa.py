#!/usr/bin/env python3
""" First basic version of the question/answer bot """

question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """
    Answers questions from a reference text

    Inputs:\\
    reference is the reference text
    """
    while True:
        question = input("Q: ")
        exits = ["exit", "quit", 'goodbye', 'bye']
        if question.lower() in exits:
            print("A: Goodbye")
            break
        answer = question_answer(question, reference)
        if answer == None or answer == "":
            answer = "Sorry, I do not understand your question."
        print(f"A: {answer}")
