#!/usr/bin/env python3
""" Module creating the list_all() function"""


def list_all(mongo_collection):
    """
    Lists all documents in a collection
    
    Inputs:
    mongo_collection - the pymongo collection object

    Returns:
    all documents in list, or empty list
    """
    return mongo_collection.find()
