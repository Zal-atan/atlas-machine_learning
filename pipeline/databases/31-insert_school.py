#!/usr/bin/env python3
""" Module creating the insert_school() function"""


def insert_school(mongo_collection, **kwargs):
    """
    Inserts a new document in a collection based on kwargs:
    
    Inputs:
    mongo_collection - the pymongo collection object

    Returns:
    the new _id
    """
    return mongo_collection.insert_one(kwargs).inserted_id
