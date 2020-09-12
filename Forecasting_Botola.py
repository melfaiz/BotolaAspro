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
            df['season'] = filename[-14:-10]
            data_selected.append(df)

# %%

data_teams = "morocco-botola-pro-teams"
data_teams_selected = []
for filename in os.listdir(data_path):
    file_path = data_path + filename
    if data_teams in filename:
        if filename[-14:-10] in str(data_years):
            df = pd.read_csv(file_path)
            data_teams_selected.append(df)
# %% [markdown]
# Concatenate and explore data over years

# %%
data = pd.concat(data_selected)
data.date_GMT = pd.to_datetime(data.date_GMT,format='%b %d %Y - %I:%M%p',errors='coerce')
data = data.reset_index()


df_teams = pd.concat(data_teams_selected)
df_teams = df_teams.reset_index()

df_teams.season = df_teams.season.apply(lambda x: x[-4:])

teams_columns_=['common_name','season','shots_on_target','goals_scored_per_match','home_advantage_percentage','points_per_game','goals_scored','average_possession','clean_sheet_percentage','leading_at_half_time_percentage','draw_at_half_time','draws_home','draws_away','losing_at_half_time_percentage','draws']
df_teams = df_teams[teams_columns_]

data2 = pd.merge(data, df_teams,  how='left', left_on=['season','home_team_name'], right_on = ['season','common_name'])
data2 = pd.merge(data2, df_teams, suffixes = ('_home', '_away'), how='left', left_on=['season','away_team_name'], right_on = ['season','common_name'])

data = data2

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
    
def draw_teams(row):
        return len( data[(data.result==0) & ((data.home_team_name==row.home_team_name) & (data.away_team_name==row.away_team_name))])+len( data[(data.result==0) & ((data.home_team_name==row.away_team_name) & (data.away_team_name==data.home_team_name))])


data['result'] = data.apply(result,axis=1)
data['home_wins'] = data.apply(home_wins,axis=1)
data['away_wins'] = data.apply(away_wins,axis=1)
data['draw_teams']=data.apply(draw_teams,axis=1)

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
columns = [
'home_wins',
'away_wins',
'home_last_5',
'away_last_5',
'home_last_all',
'away_last_all',
'shots_on_target_home', 'goals_scored_per_match_home',
'home_advantage_percentage_home', 'points_per_game_home',
'goals_scored_home', 'average_possession_home',
'clean_sheet_percentage_home', 'leading_at_half_time_percentage_home',
'draw_at_half_time_home', 'shots_on_target_away',
'goals_scored_per_match_away', 'home_advantage_percentage_away',
'points_per_game_away', 'goals_scored_away', 'average_possession_away',
'clean_sheet_percentage_away', 'leading_at_half_time_percentage_away',
'draw_at_half_time_away',
'draw_teams',
'losing_at_half_time_percentage_home',
'losing_at_half_time_percentage_away',
'draws_home_home',
'draws_home_away',
'draws_away_home',
'draws_away_away',
'draws_away',
'draws_home',
'result'
]

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


from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


clf_decision_tree = tree.DecisionTreeClassifier(max_depth=7)

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
_ = tree.plot_tree(clf_decision_tree, 
                feature_names=X.columns,  
                class_names=['draw','home','away'],
                filled=True)

# %%

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

clf_decision_tree.feature_importances_  

model = SelectFromModel(clf_decision_tree, prefit=True)
X_new = model.transform(X)
X_new.shape   
# %%
