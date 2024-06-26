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


###   Global variables   ###
#userRatingMatrix => merge users and their ratings
uRM = pd.merge(users_description, ratings_description, on='userID')

#userRatingMoviesMatrix => merge users+ratings on the movies they watched
uRMM = pd.merge(uRM, movies_description, on='movieID')

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
#Fill in lost columns
userMovie = userMovie.reindex(pd.RangeIndex(1, max(userMovie.columns) + 1), axis='columns')

normal_mU = uf.normalized_data(moviesUser)
normal_uM = uf.normalized_data(userMovie)
overall_movie_mean = np.nanmean(moviesUser)


#####
##
## COLLABORATIVE FILTERING
##
#####

def predict_collaborative_filtering(movies, users, ratings, predictions):
  # TO COMPLETE

  #6040 users & 3706 movies !!!
  #Item-Item collaborative matrix = pearson(userMovie).shape = (3706, 3706)
  #User-User collaborative matrix = pearson(movieUser).shape = (6040,6040)

  ##Item-Item
  correlation_item = uf.pearson(userMovie)
  nn_item = uf.threshold(0.8, 50, correlation_item)
  item_ratings = uf.ratingItem(predictions, correlation_item, nn_item, userMovie, normal_uM, overall_movie_mean)


  ##User-User
  # correlation_user = uf.pearson(moviesUser)
  # nn_user = uf.threshold(0.9, 10, correlation_user)
  # user_ratings = uf.ratingUser(predictions, correlation_user, nn_user, moviesUser, normal_mU, overall_movie_mean).values
  

  #Create the IDs that we will pass on to the submission.csv file
  ids = np.arange(1, len(predictions) + 1)

  #We will insert the IDs column to the left of all the ratings
  predict_score = np.vstack((ids, item_ratings)).transpose()

  return predict_score


#####
##
## LATENT FACTORS
##
#####
    
def predict_latent_factors(movies, users, ratings, predictions):
  ## TO COMPLETE

  #Handle NaNs =>  fill it with zeros
  X = normal_uM.fillna(0)
  # print("\nnormal_uM filled w zeros ==>> \n\n" , X.shape)

  u, s, vh = np.linalg.svd(X)

  sQuared = (s*s).tolist()
  total_energy = np.sum(sQuared)
  econ_energy = 0.8*total_energy

  temp = 0
  lf = 0
  #Find index (=lf) where the most energy we want is conserved
  #We don't use this loop if lf = set to specific number
  # for i in range(len(sQuared)):
  #     temp+=sQuared[i]
  #     if(temp >= econ_energy):
  #         lf = i
  #         break

  lf = 50

  Q = u[:, :lf]
  sigma = np.diag(s[:lf])
  Vh = vh[:lf, :]
  Pt = np.dot(sigma, Vh)   #np.allclose(Pt, (sigma @ Vh)) => True

  # X_econ = (Q @ Pt)

  all_ratings = uf.SVDrating(predictions, userMovie, Q, Pt, overall_movie_mean)

  #Create the IDs that we will pass on to the submission.csv file
  ids = np.arange(1, len(predictions) + 1)
  pred_SVD = np.vstack((ids, all_ratings)).transpose()

  return pred_SVD
    
    
#####
##
## FINAL PREDICTORS
##
#####

def predict_final(movies, users, ratings, predictions):
  # cf = predict_collaborative_filtering(movies, users, ratings, predictions)
  # lf = predict_latent_factors(movies, users, ratings, predictions)

  return predict_latent_factors(movies, users, ratings, predictions)

  # return np.round_(((cf + lf) / 2), 2)


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
predictions = predict_final(movies_description, users_description, ratings_description, predictions_description)

#Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
  #Formates data
  predictions = [[int(row[0]), row[1]] for row in predictions]
  predictions = [map(str, row) for row in predictions]
  predictions = [','.join(row) for row in predictions]
  predictions = 'Id,Rating\n'+'\n'.join(predictions)
  
  #Writes it dowmn
  submission_writer.write(predictions)