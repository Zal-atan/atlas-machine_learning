#!/usr/bin/env python3
"""Script that displays the specific information about the nearest upcoming
SpaceX launch"""

from requests import get

if __name__ == "__main__":

    LAUNCHES = 'https://api.spacexdata.com/v5/launches/'

    response = get(LAUNCHES + 'upcoming')
    upcoming_launch_json = response.json()

    # upcoming_launch_json
    # Make list of each piece of launch info from first API
    dates_list = [launch['date_local'] for launch in upcoming_launch_json]
    names_list = [launch['name'] for launch in upcoming_launch_json]
    rockets_list = [launch['rocket'] for launch in upcoming_launch_json]
    pads_list = [launch['launchpad'] for launch in upcoming_launch_json]

    # Find the list position of closest upcoming launch
    i = dates_list.index(min(dates_list))

    # Get name and date from first API info
    name = names_list[i]
    date = dates_list[i]

    # Get info about the rocket
    ROCKETS = 'https://api.spacexdata.com/v4/rockets/'
    # response = get(ROCKETS + ':' + rockets_list[i])
    response = get(ROCKETS)

    rockets_json = response.json()
    rocket = rockets_json[1]['name']
    # rockets_list[i]
    # upcoming_launch_json
    rocket

    # Get info about the launchpad
    LAUNCHPAD = 'https://api.spacexdata.com/v4/launchpads/'

    response = get(LAUNCHPAD + pads_list[i])
    pads_json = response.json()
    pad = pads_json['name']
    locality = pads_json['locality']

    print(f"{name} ({date}) {rocket} - {pad} ({locality})")
