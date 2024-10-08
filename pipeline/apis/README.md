# This is a README for the APIs repo.

### In this repo we will practicing using APIs for applications of Data Science
<br>

### Author - Ethan Zalta
<br>


# Tasks
### There are 5 tasks in this project

## Task 0
* By using the Swapi API, create a method that returns the list of ships that can hold a given number of passengers:

    * Prototype: def availableShips(passengerCount):
    * Don’t forget the pagination
    * If no ship available, return an empty list.

## Task 1
* By using the Swapi API, create a method that returns the list of names of the home planets of all sentient species.

    * Prototype: def sentientPlanets():
    * Don’t forget the pagination
    * sentient type is either in the classification or designation attributes.

## Task 2
* By using the GitHub API, write a script that prints the location of a specific user:

    * The user is passed as first argument of the script with the full API URL, example: ./2-user_location.py https://api.github.com/users/holbertonschool
    * If the user doesn’t exist, print Not found
    * If the status code is 403, print Reset in X min where X is the number of minutes from now and the value of X-Ratelimit-Reset
    * Your code should not be executed when the file is imported (you should use if __name__ == '__main__':)

## Task 3
* By using the (unofficial) SpaceX API, write a script that displays the first launch with these information:

    * Name of the launch
    * The date (in local time)
    * The rocket name
    * The name (with the locality) of the launchpad
    * Format: <launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)

we encourage you to use the date_unix for sorting it - and if 2 launches have the same date, use the first one in the API result.

Your code should not be executed when the file is imported (you should use if __name__ == '__main__':)

## Task 4
* By using the (unofficial) SpaceX API, write a script that displays the number of launches per rocket.

    * Use this https://api.spacexdata.com/v4/launches to make request
    * All launches should be taken into consideration
    * Each line should contain the rocket name and the number of launches separated by : (format below in the example)
    * Order the result by the number launches (descending)
    * If multiple rockets have the same amount of launches, order them by alphabetic order (A to Z)
    * Your code should not be executed when the file is imported (you should use if __name__ == '__main__':)

