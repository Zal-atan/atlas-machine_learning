#!/usr/bin/env python3
""" Module creating the schools_by_topic() function"""


def schools_by_topic(mongo_collection, topic):
    """
    Returns the list of school having a specific topic

    Inputs:
    mongo_collection - the pymongo collection object\\
    topic -  (string) will be topic searched

    Returns:
    list of schools having a specific topic
    """
    return mongo_collection.find({"topics": topic})
