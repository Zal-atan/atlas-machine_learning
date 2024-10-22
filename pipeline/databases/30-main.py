#!/usr/bin/env python3
""" 30-main """
from pymongo import MongoClient
list_all = __import__('30-all').list_all

if __name__ == "__main__":
    client = MongoClient('mongodb+srv://Cluster24341:VUhpS2NZQVp5@cluster24341.kdz4x.mongodb.net/')
    school_collection = client.my_db.school
    schools = list_all(school_collection)
    for school in schools:
        print("[{}] {}".format(school.get('_id'), school.get('name')))
