#!/usr/bin/env python3
"""Script that prints the location of a specific user from github"""
from requests import get
import time
import sys

if __name__ == "__main__":
    args = sys.argv

    usage = "Usage: ./2-user_location.py https://api.github.com/users/<user>"
    if len(args) != 2:
        print(usage)
        quit()

    site = args[1]

    response = get(site)

    user_json = response.json()
    # print(user_json)

    # Correct Output
    if response.status_code == 200:
        print(user_json['location'])

    # User not found
    elif response.status_code == 404:
        print("Not found")

    # Too many requests
    elif response.status_code == 403:
        timer = response.headers['X-RateLimit-Reset']
        now = int(time.time())
        difference = (int(timer) - now) // 60
        print(f"Reset in {difference} min")
