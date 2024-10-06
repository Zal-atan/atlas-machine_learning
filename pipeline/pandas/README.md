# This is a README for the Pandas repo.

### In this repo we will practicing Pandas for applications of Data Science
<br>

### Author - Ethan Zalta
<br>


# Tasks
### There are 15 tasks in this project

## Task 0
* Write a function def from_numpy(array): that creates a pd.DataFrame from a np.ndarray:

    * array is the np.ndarray from which you should create the pd.DataFrame
    * The columns of the pd.DataFrame should be labeled in alphabetical order and capitalized. There will not be more than 26 columns.
    * Returns: the newly created pd.DataFrame

## Task 1
* Write a python script that created a pd.DataFrame from a dictionary:

    * The first column should be labeled First and have the values 0.0, 0.5, 1.0, and 1.5
    * The second column should be labeled Second and have the values one, two, three, four
    * The rows should be labeled A, B, C, and D, respectively
    * The pd.DataFrame should be saved into the variable df

## Task 2
* Write a function def from_file(filename, delimiter): that loads data from a file as a pd.DataFrame:

    * filename is the file to load from
    * delimiter is the column separator
    * Returns: the loaded pd.DataFrame

## Task 3
* Complete the script below to perform the following:

    * Rename the column Timestamp to Datetime
    * Convert the timestamp values to datatime values
    * Display only the Datetime and Close columns

## Task 4
* Complete the following script to take the last 10 rows of the columns High and Close and convert them into a numpy.ndarray:

## Task 5
* Complete the following script to slice the pd.DataFrame along the columns High, Low, Close, and Volume_BTC, taking every 60th row:

## Task 6
* Complete the following script to alter the pd.DataFrame such that the rows and columns are transposed and the data is sorted in reverse chronological order:


## Task 7
* Complete the following script to sort the pd.DataFrame by the High price in descending order:


## Task 8
* Complete the following script to remove the entries in the pd.DataFrame where Close is NaN:


## Task 9
* Complete the following script to fill in the missing data points in the pd.DataFrame:

    * The column Weighted_Price should be removed
    * missing values in Close should be set to the previous row value
    * missing values in High, Low, Open should be set to the same row’s Close value
    * missing values in Volume_(BTC) and Volume_(Currency) should be set to 0


## Task 10
* Complete the following script to index the pd.DataFrame on the Timestamp column:


## Task 11
* Complete the following script to index the pd.DataFrames on the Timestamp columns and concatenate them:

    * Concatenate the start of the bitstamp table onto the top of the coinbase table
    * Include all timestamps from bitstamp up to and including timestamp 1417411920
    * Add keys to the data labeled bitstamp and coinbase respectively


## Task 12
* Based on 11-concat.py, rearrange the MultiIndex levels such that timestamp is the first level:

    * Concatenate th bitstamp and coinbase tables from timestamps 1417411980 to 1417417980, inclusive
    * Add keys to the data labeled bitstamp and coinbase respectively
    * Display the rows in chronological order


## Task 13
* Complete the following script to calculate descriptive statistics for all columns in pd.DataFrame except Timestamp:


## Task 14
* Complete the following script to visualize the pd.DataFrame:

    * The column Weighted_Price should be removed
    * Rename the column Timestamp to Date
    * Convert the timestamp values to date values
    * Index the data frame on Date
    * Missing values in Close should be set to the previous row value
    * Missing values in High, Low, Open should be set to the same row’s Close value
    * Missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
    * Plot the data from 2017 and beyond at daily intervals and group the values of the same day such that:

        * High: max
        * Low: min
        * Open: mean
        * Close: mean
        * Volume(BTC): sum
        * Volume(Currency): sum

