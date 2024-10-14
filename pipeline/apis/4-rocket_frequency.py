#!/usr/bin/env python3
"""Script that displays the number of times each rocket has been launched, in
order of most to least launches"""

from requests import get


if __name__ == "__main__":
    # Get info about the rocket
    ROCKETS = 'https://api.spacexdata.com/v4/rockets/'

    rockets_dict = {}

    response = get(ROCKETS)
    rockets_json = response.json()

    for rocket in rockets_json:
        rockets_dict[rocket['id']] = rocket['name']

    # Catalog each launch
    LAUNCHES = 'https://api.spacexdata.com/v5/launches/'

    launch_dict = {}

    response = get(LAUNCHES)
    launch_json = response.json()

    for launch in launch_json:
        name = rockets_dict[launch['rocket']]
        launch_dict[name] = launch_dict.get(name, 0) + 1
    # print(launch_dict)

    sorted_launches = sorted(launch_dict.items(), key=lambda x: -x[1])
    # sorted_launches
    for num in sorted_launches:
        print(num[0] + ': ' + str(num[1]))
