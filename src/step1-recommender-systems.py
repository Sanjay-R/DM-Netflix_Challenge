import numpy as np
import pandas as pd
import os.path
from random import randint
import utilFunctions as uf

# -*- coding: utf-8 -*-
"""
### NOTES
This file is an example of what your code should look like. It is written in Python 3.6.
To know more about the expectations, please refer to the guidelines.
"""

#####
##
## DATA IMPORT
##
#####

#Where data is located
movies_file = './data/movies.csv'
users_file = './data/users.csv'
ratings_file = './data/ratings.csv'
predictions_file = './data/predictions.csv'
submission_file = './data/submission.csv'


# Read the data using pandas
movies_description = pd.read_csv(movies_file, delimiter=';', dtype={'movieID':'int', 'year':'int', 'movie':'str'}, names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';', dtype={'userID':'int', 'gender':'str', 'age':'int', 'profession':'int'}, names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';', dtype={'userID':'int', 'movieID':'int', 'rating':'int'}, names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'], header=None)

#####
##
## COLLABORATIVE FILTERING
##
#####

def predict_collaborative_filtering(movies, users, ratings, predictions):
    # TO COMPLETE

    #userRatingMatrix => merge users and their ratings
    uRM = pd.merge(users, ratings, on='userID')

    #userRatingMoviesMatrix => merge users+ratings on the movies they watched
    uRMM = pd.merge(uRM, movies, on='movieID')

    #moviesUser => matrix which sets userID on the rows and movieID on the column. 
    #This is used for User-User CF
    #The ratings are filled in as values
    moviesUser = uRMM.pivot(index='movieID', columns='userID', values='rating')
    #Now make sure to fill in lost rows
    moviesUser = moviesUser.reindex(pd.RangeIndex(1, moviesUser.index.max() + 1))



    #userMovie => matrix which sets movieID on the rows and userID on the column.
    #This is used for Item-Item CF
    #The ratings are filled in as values
    userMovie = uRMM.pivot(index='userID', columns='movieID', values='rating')

    

    #User-User collaborative matrix
    utilUser = uf.pearson(moviesUser)

    #Item-Item collaborative matrix
    utilItem = uf.pearson(userMovie)

    nn = uf.threshold(0.8, 10, utilUser)


    #These are all the ratings we get for all (userID, movieID) pair passed on from predictions.csv
    all_ratings = uf.rating(predictions, utilUser, nn, moviesUser).values

    #Create the IDs that we will pass on to the submission.csv file
    ids = np.arange(1, len(predictions) + 1)

    #We will insert the IDs column to the left of all the ratings
    predict_score = np.vstack((ids, all_ratings)).transpose()

    return predict_score


#####
##
## LATENT FACTORS
##
#####
    
def predict_latent_factors(movies, users, ratings, predictions):
    ## TO COMPLETE

    pass
    
    
#####
##
## FINAL PREDICTORS
##
#####

def predict_final(movies, users, ratings, predictions):
  ## TO COMPLETE

  pass


#####
##
## RANDOM PREDICTORS
## //!!\\ TO CHANGE
##
#####
    
#By default, predicted rate is a random classifier
def predict_random(movies, users, ratings, predictions):
    number_predictions = len(predictions)

    return [[idx, randint(1, 5)] for idx in range(1, number_predictions + 1)]


#####
##
## SAVE RESULTS
##
#####    

## //!!\\ TO CHANGE by your prediction function
# predictions = predict_random(movies_description, users_description, ratings_description, predictions_description)
predictions = predict_collaborative_filtering(movies_description, users_description, ratings_description, predictions_description)

#Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
    #Formates data
    predictions = [[int(row[0]), row[1]] for row in predictions]
    predictions = [map(str, row) for row in predictions]
    predictions = [','.join(row) for row in predictions]
    predictions = 'Id,Rating\n'+'\n'.join(predictions)
    
    #Writes it dowmn
    submission_writer.write(predictions)