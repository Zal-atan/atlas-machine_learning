#!/usr/bin/env python3
""" Module creating the update_topics() function"""


def update_topics(mongo_collection, name, topics):
    """
    Changes all topics of a school document based on the name:

    Inputs:
    mongo_collection - the pymongo collection object\\
    name - (string) will be the school name to update\\
    topics - (list of strings) will be the list of topics approached
    in the school

    Returns:
    none
    """
    return mongo_collection.update_many({
        "name": name
    },
        {
        "$set": {
            "name": name,
            "topics": topics
        }
    })
