#!/usr/bin/env python3
""" Module with basic QA bot exit feature """


while True:
    question = input("Q: ")
    exits = ["exit", "quit", 'goodbye', 'bye']
    if question.lower() in exits:
        print("A: Goodbye")
        break
    print("A: ")
