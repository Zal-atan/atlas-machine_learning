#!/usr/bin/env python3
""" Module containing the availableShips() function"""
from requests import get


BASE_API = "https://swapi-api.hbtn.io/api/"


def availableShips(passengerCount):
    """
    Method that returns the list of ships that can hold a
    given number of passengers

    Inputs:
    passengerCount - int - number of passengers

    Returns:
    List of ships which can carry passengerCount number of passengers
    """
    ships_list = []

    response = get(BASE_API + "starships/")

    ships_json = response.json()
    # print(ships_json)

    more_ships = True
    while more_ships:
        page = 1
        while ships_json['next']:
            response = get(BASE_API + "starships/" + f"?page={page}")
            ships_json = response.json()
            for ship in ships_json['results']:
                ship['passengers'] = ship['passengers'].replace(',', '')
                if ship['passengers'] == "unknown":
                    continue
                if ship['passengers'] == "n/a":
                    continue
                if int(ship['passengers']) >= passengerCount:
                    ships_list.append(ship['name'])

            page += 1
            # print(ships_list)

        more_ships = False

    return ships_list
