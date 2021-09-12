#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from array import *

data = pd.read_csv("spotify.csv",index_col=[0])

data.isnull().any().sum()

## array and matrix are working in the same fashion
X = data.values

# converting a dataframe into a sparse matrix (very big matrix,min100) or an array
X[0][0]

#  ## Performing NMF - Non Negative Matrix Factorization

# Matrix factorization - decomposing a large matrix into 2 smaller matrices

from sklearn.decomposition import NMF
nmf = NMF(n_components=100,max_iter=1000,random_state=200)

# # above line - n_components i.e. no of rows and columns = 100(can be anything but less than no of features)
#              - max_iter = 6*4=24, 24 can have many factors, so no. of factiorizations to get the best value. 
#                 1000 different factorizations
#               - random state - random different no chosen
nmf.fit(X)

user_matrix = nmf.transform(X)

pd.DataFrame(user_matrix)
user_matrix.shape
song_matrix = nmf.components_
song_matrix.shape
pd.DataFrame(song_matrix.T)
# ## Steps in building a  simple recommendation system
#     (For ease of understanding,let us recommend songs to user1 (located at row 0) in user matrix
#     Step #1 Find the direct distance between the user1 and the rest of the 999 users using the total no of times a song is being heard
#     Step #2 Find out the closest 5 users and recommend the top 10 songs that they are listening to user1

# step 1 explaination : if a certain set of users are listening to similar songs same no of times then they will form a cluster.
# our goal here is to identify which users listening pattern is similar to user1,once we have found out the closest 5  users we can recommend top 10 songs that these closest users are listening and recommend it to user1
# ## Finding direct euclidean dist between users
# euclidean dist = sqrt((x2-x1)^2+(y2-y1)^2+......)

def get_dist(userA,userB):
    return pow(sum((pow(userA[x]-userB[x],2)for x in range (len(userA)))),0.5)    

get_dist(user_matrix[0],user_matrix[1])

# ## Finding direct distance between user1 and all other users (999)
def get_dist_from_user(base_user_index,users_matrix):
    import sys
    distance = []
    for i in range (len(users_matrix)):   # will run 1000 times
        if base_user_index != i:          # dist between self will be 0.so we prevent it
            distance.append(get_dist(users_matrix[base_user_index],users_matrix[i]))
        else:
            distance.append(sys.float_info.max)        # if self, we append a large float value
    return distance

user_Dist = get_dist_from_user(0,user_matrix)

user_Dist

user_index = np.argsort(user_Dist)[0:5]   # gives sorted data index in ascending order and give 1st 5

user_index

# ## Finding the songs heard by the closest users and recommending top 10 songs to user1

for i in user_index:
    print('Songs heard by user in index position',i,' are: ')
    temp = pd.DataFrame(data.iloc[i])   # gives the exact loc of columns
    print(temp[temp.values!=0].index)
    #print(temp[temp.values!=0])
def top_n_songs(closest_user_index,dataframe,no_of_songs):
    temp_df = dataframe.iloc[closest_user_index]   # original dataframe
    dict1 = temp_df.max()   # song heard the max no of times by closest user

temp_df = data.iloc[user_index]

temp_df

temp_df.max()

sum_songs = temp_df.sum(axis=0)

sum_songs

sort_songs = sum_songs.sort_values(ascending=False)

sort_songs.head(10)

def top_n_songs(closest_user_index,dataframe,no_of_songs):
    temp_df = dataframe.iloc[closest_user_index]   # original dataframe
    sum_of_songs = temp_df.sum(axis=0)
    sort_songs = sum_of_songs.sort_values(ascending=False)[0:no_of_songs]
    return sort_songs.index

top_n_songs(user_index,data,10)

## lenghthy process, if database changes, lengthy and complex model

## use of unsupervised ml algo used for this purpose
## it does not produce any final result.
## pca is also an unsupervised ml algo

# find cluster of song.basis the no of times the song is heard.
# and we find which is the 1 song the user will hear


#K-means clustering
# k = centroid (not the no of neighbors) ie no of dense clusters in the data
# elbow technique/ curve is used to determine k

# ## Recommendation system using K Means clustering

from sklearn.cluster import KMeans

within_cluster_mean_sq_error = {}
for k in range (3,40):
    kmeans = KMeans(n_clusters=k,max_iter=1000).fit(song_matrix.T)    #max iter better as much as possible.keeping less leads to underfittiing,more will not lead to overfitting.without transpose, only time for 100 it is taking. with transpose it will consider 5000
    within_cluster_mean_sq_error[k] = kmeans.inertia_   #accuracy stored

#within_cluster_mean_sq_error

#within cluster sum of square is same as accuracy. inertia will try to predict if the  cluster and centroid is proper or not.accuracy is measured by MSE.how every data point is distributed around the centroid is given by MSE.as k increases. mse decreases.affter a particular point, mse decreases less significantly.


plt.figure()
plt.plot(list(within_cluster_mean_sq_error.keys()),list(within_cluster_mean_sq_error.values()))
# ## Recommending songs based on what song a particular user has heard,and not on basis of how close the users are but on the no.of times each song is heard.

def ret_songs_from_a_cluster(data,songs_matrix,song_name,n_recommendation): #defining the function and taking the following inputs from user : main data,decomposed song matrix,base song,noof recommendations
    k_means = KMeans(n_clusters=27,max_iter=1000).fit(songs_matrix) #making clusters of the songs based on the decomposed no of times each song is heard
    all_songs_in_cluster = list(k_means.predict(songs_matrix)) #finding the cluster to which each song belongs to.
    
    index_of_song = data.columns.to_list().index(song_name) # getting the index of the base song from main data
    song_vector = songs_matrix[index_of_song] # copying the decomposed list ening value from the song matrix and index of the base song
    #finding out to which cluster does the base son belongs 
    songs_in_selected_cluster = [x for x in range(len(all_songs_in_cluster)) if all_songs_in_cluster[x]==k_means.predict([song_vector])] 
    
    song_cluster = song_matrix[songs_in_selected_cluster]  #isolating the matching cluster & get the original song names
    
    knn = NearestNeighbors(n_neighbors=n_recommendation) 
    knn.fit(song_cluster) # fit knn algo on the song_cluster with n_neighbors being identified/recommended
    recommend_songs = knn.kneighbors([song_matrix[index_of_song]])[1]  # since we are working on the decomposed values we do not pass sng_name directly #identify closest neighbors & returning the indexes
    final_song_index = list(recommend_songs[0])
    final_song_list = list(data.columns[final_song_index])
    return final_song_index,final_song_list


data.columns.to_list().index('song_5')
ret_songs_from_a_cluster(data,song_matrix,'song_5',10)
