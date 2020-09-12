# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Predicting a match winner 
# (Multiclass classification problem)
# %% [markdown]
# Importing libraries

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# Writing paths and variables

# %%
data_path = "Botola_Forecasting/"
data_type = "morocco-botola-pro-matches"
data_years = [2017,2018,2019,2020]


# %% [markdown]
# Merging years of data

# %%
data_selected = []
for filename in os.listdir(data_path):
    file_path = data_path + filename
    if data_type in filename:
        if filename[-14:-10] in str(data_years):
            df = pd.read_csv(file_path)
            data_selected.append(df)

# %% [markdown]
# Concatenate and explore data over years

# %%
data = pd.concat(data_selected)



data.date_GMT = pd.to_datetime(data.date_GMT,format='%b %d %Y - %I:%M%p',errors='coerce')

data = data.reset_index()
# %%

data.head()

# %% [markdown]
# Different teams that played in the selected years

# %%
teams = pd.concat([data.away_team_name,data.home_team_name]).unique()
for team in teams:
    print("-> ",team)

# %% [markdown]
# Team with most home wins

# %%
# data[ (data.home_team_goal_count > data.away_team_goal_count) ].home_team_name.value_counts().plot.bar()


# # %%
# data[ (data.home_team_goal_count < data.away_team_goal_count) ].away_team_name.value_counts().plot.bar()


# # %%
# data[ (data.home_team_goal_count > data.away_team_goal_count) ].home_team_name.value_counts().plot.bar()


# %%
def result(row):
    if row.home_team_goal_count > row.away_team_goal_count:
        return 1
    elif row.home_team_goal_count < row.away_team_goal_count:
        return 2
    else:
        return 0

def away_wins(row):
    home_name = row.home_team_name
    away_name = row.away_team_name

    return len(  data[  ( (data.result == 1) & (data.home_team_name == away_name) & (data.away_team_name == home_name)  ) 
                | ((data.result == 2) & (data.home_team_name == home_name) & (data.away_team_name == away_name)) ] )

def home_wins(row):
    home_name = row.home_team_name
    away_name = row.away_team_name

    return len(  data[  ( (data.result == 1) & (data.home_team_name == home_name) & (data.away_team_name == away_name)  ) 
                | ((data.result == 2) & (data.home_team_name == away_name) & (data.away_team_name == home_name)) ] )

def last_matches(date,team,n):
    df = data[ ((data.home_team_name == team) | (data.away_team_name == team)) & (data.date_GMT < date) & (data.status == 'complete') ][-n:]
    return len( df[ (df.home_team_name == team) & (df.result == 1) ] ) + len( df[ (df.away_team_name == team) & (df.result == 2) ] ) 
    



data['result'] = data.apply(result,axis=1)
data['home_wins'] = data.apply(home_wins,axis=1)
data['away_wins'] = data.apply(away_wins,axis=1)


# %%

data['home_last_5'] = 0
data['away_last_5'] = 0
for i in range(len(data)):
    h = last_matches(data.iloc[i].date_GMT,data.iloc[i].home_team_name,10)
    a = last_matches(data.iloc[i].date_GMT,data.iloc[i].away_team_name,10)
    data.at[i,'home_last_5'] = h
    data.at[i,'away_last_5'] = a


# %%
data['home_last_all'] = 0
data['away_last_all'] = 0
for i in range(len(data)):
    h = last_matches(data.iloc[i].date_GMT,data.iloc[i].home_team_name,0)
    a = last_matches(data.iloc[i].date_GMT,data.iloc[i].away_team_name,0)
    data.at[i,'home_last_all'] = h
    data.at[i,'away_last_all'] = a


# %%
# mask = (data.home_team_name == 'Raja Casablanca') & (data.status == 'complete')
# data[mask]


# %%

# labels = 'Home wins', 'Away wins', 'Draw'


# plt.figure(figsize=(12,7))
# plt.pie(data.result.value_counts(), labels=labels, autopct='%1.1f%%',
#         shadow=True, startangle=90)
# plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# plt.show()

# data.result.value_counts()


# %%
columns = ['Pre-Match PPG (Home)','Pre-Match PPG (Away)','home_wins','away_wins','home_last_5','away_last_5','home_last_all','away_last_all','result']

data_learning = data[data.status == 'complete']
data_learning = data_learning[columns]


X = data_learning.drop(['result'],1)
y = data_learning.result

X.head()


# %%
# from sklearn.preprocessing import scale

# num_columns = ['Pre-Match PPG (Home)','Pre-Match PPG (Away)','home_wins','away_wins']
# for col in num_columns:
#     X[col] = scale(X[col])

# X.head()



# %%
#Using Pearson Correlation
import seaborn as sns
plt.figure(figsize=(12,10))
cor = data_learning.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# %%

#we want continous vars that are integers for our input data, so lets remove any categorical vars
# def preprocess_features(X):
#     ''' Preprocesses the football data and converts catagorical variables into dummy variables. '''
    
#     # Initialize new output DataFrame
#     output = pd.DataFrame(index = X.index)

#     # Investigate each feature column for the data
#     for col, col_data in X.iteritems():

#         # If data type is categorical, convert to dummy variables
#         if col_data.dtype == object:
#             col_data = pd.get_dummies(col_data, prefix = col)
                    
#         # Collect the revised columns
#         output = output.join(col_data)
    
#     return output

# X = preprocess_features(X)

# %%

# from sklearn import preprocessing

# le = preprocessing.LabelEncoder()

# le.fit(teams)

# home_teams = le.transform( X.home_team_name)
# away_teams = le.transform( X.away_team_name)

# X['home_team_name'] = home_teams
# X['away_team_name'] = away_teams

# %%

from sklearn.model_selection import train_test_split

# Shuffle and split the dataset into training and testing set.
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3,
                                                    random_state = 2,
                                                    stratify = y)


# %%

#for measuring training time
from time import time 
# F1 score (also F-score or F-measure) is a measure of a test's accuracy. 
#It considers both the precision p and the recall r of the test to compute 
#the score: p is the number of correct positive results divided by the number of 
#all positive results, and r is the number of correct positive results divided by 
#the number of positive results that should have been returned. The F1 score can be 
#interpreted as a weighted average of the precision and recall, where an F1 score 
#reaches its best value at 1 and worst at 0.
from sklearn.metrics import f1_score

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print ("Trained model in {:.4f} seconds".format(end - start))

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    
    end = time()
    # Print and return results
    print ("Made predictions in {:.4f} seconds.".format(end - start))
    
    return f1_score(target, y_pred, pos_label='1',average='micro'), sum(target == y_pred) / float(len(y_pred))


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print ("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    f1, acc = predict_labels(clf, X_train, y_train)
    print ("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1 , acc))
    
    f1, acc = predict_labels(clf, X_test, y_test)
    print ("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1 , acc))

    y_pred = clf.predict(X_test)

    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(confusion_matrix(y_test,y_pred), annot=True,cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# %%

from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


clf_decision_tree = tree.DecisionTreeClassifier(max_depth=4)

clf_decision_tree.fit(X_train, y_train)

y_pred = clf_decision_tree.predict(X_test)

# ACCURACY

accuracy = sum ( y_pred == y_test) / len(y_test)
print("Test accuracy : ",accuracy)
# CLASSIFICATION REPORT

''' 
precision: ration of true positives



'''

print('\n'+classification_report(y_test, y_pred, target_names=['draw','home','away']))


# CONFUSION MATRIX

figure = plt.figure(figsize=(8, 8))
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# DECISION TREE PLOT

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf_D, 
                feature_names=X.columns,  
                class_names=['draw','home','away'],
                filled=True)

# %%
