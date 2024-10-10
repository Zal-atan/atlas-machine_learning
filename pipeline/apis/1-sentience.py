#!/usr/bin/env python3
""" Module containing the sentientPlanet() function"""
from requests import get

BASE_API = "https://swapi-api.hbtn.io/api/"


def sentientPlanets():
    """
    Returns the list of names of the home planets of all sentient species

    Returns:
    List of home planets of all sentient species
    """

    planets_list = []
    species_hw_list = []

    # Find all planets
    response = get("https://swapi-api.hbtn.io/api/planets/")

    planets_json = response.json()

    page = 1
    while planets_json['next']:
        response = get(BASE_API + "planets/" + f"?page={page}")
        planets_json = response.json()
        for planet in planets_json['results']:
            planets_list.append(planet['name'])

        page += 1

    # Find all species homeworlds
    response = get(BASE_API + "species/")

    species_json = response.json()

    page = 1
    while species_json['next']:
        response = get(BASE_API + "species/" + f"?page={page}")
        species_json = response.json()
        for species in species_json['results']:
            designation = species['designation'] == 'sentient'
            classification = species['classification'] == 'sentient'
            if designation or classification:
                if species['homeworld'] == "None":
                    continue
                try:
                    world_response = get(species['homeworld'])
                except BaseException:
                    continue
                world_json = world_response.json()
                species_hw_list.append(world_json['name'])

        page += 1

    final_planet_list = []

    for planet in planets_list:
        if planet in species_hw_list:
            final_planet_list.append(planet)

    # for planet in planets_list:
    #     print(planet)
    # response = get(BASE_API + "species/")

    # species_json = response.json()
    # for species in species_hw_list:
    #     print(species)
    return final_planet_list
