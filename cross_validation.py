import pandas as pd
import numpy as np

from sklearn import (
    metrics,
    model_selection,
    svm,
    ensemble
)

# Read CSVs from memory and remove duplicates
dataset_accidents = pd.read_csv(
    'data/accidents.csv',
    header=0
)
dataset_accidents = dataset_accidents.drop_duplicates()

dataset_vehicles = pd.read_csv(
    'data/vehicles.csv',
    header=0
)
dataset_vehicles = dataset_vehicles.drop_duplicates()

# There were some accidents placed in the coast of Morocco,
# we suppose all of them are noise and they get discarded.
# Moreover, they are all from the majority class, so, it's
# not too problematic to load information with this drop
dataset_accidents = dataset_accidents[
    dataset_accidents['latitude'] >= 40
]

# The information about location is redundant with the police_force
# column, since the whole country is divided in regions controlled
# by each force.
dataset_accidents = dataset_accidents.drop(
    ['location_easting_osgr', 'location_northing_osgr',
     'lsoa_of_accident_location', 'latitude', 'longitude'],
    axis=1
)

# Variables police_force, local_authority_highway and
# local_authority_district represent the same information, so
# only the most concise information is kept
dataset_accidents = dataset_accidents.drop(
    ['local_authority_district', 'local_authority_highway'],
    axis=1
)

# Temporal information converted into more relevant data
dataset_accidents['weekday'] = pd.to_datetime(
    dataset_accidents['date']
).dt.weekday_name

dataset_accidents['weekend'] = (dataset_accidents['weekday'].isin(
    ['Friday', 'Saturday', 'Sunday']
))*1

dataset_accidents['day_period'] = pd.to_datetime(
    dataset_accidents['time']
).dt.hour

dataset_accidents['day_period'] = pd.cut(
    dataset_accidents['day_period'], bins=[0, 7, 9, 13, 16, 20, 24], right=False
)

dataset_accidents = dataset_accidents.drop(['date', 'time', 'weekday'], axis=1)

# Variables with too much invalid or non computed values
dataset_accidents = dataset_accidents.drop(
    [
        'pedestrian_crossing-human_control',
        'pedestrian_crossing-physical_facilities',
        'carriageway_hazards'
    ],
    axis=1
)

# Urban and rural into boolean variable
dataset_accidents['urban_area'] = 1*(
    dataset_accidents['urban_or_rural_area'] == 'Urban'
)
dataset_accidents = dataset_accidents.drop('urban_or_rural_area', axis=1)

# Class is more representative than the number of the road
dataset_accidents = dataset_accidents.drop(
    ['1st_road_number', '2nd_road_number'], axis=1
)

# MERGING BOTH DATASETS USING PRIMARY KEY TO HAVE INFORMATION
# FROM TWO SOURCES
dataset = dataset_accidents.merge(dataset_vehicles, on='accident_id')

# Drop accident_id since it's irrelevant after merging
dataset = dataset.drop('accident_id', axis=1)

# We delete the sex of the driver on purpose since we prefer not to
# take this information into account
dataset = dataset.drop('Sex_of_Driver', axis=1)

# A code identifier about the car is not important to predict the severity of an accident
dataset = dataset.drop('Vehicle_Reference', axis=1)

# Percentage of serious accidents is approx constant independently of IMD value, so discarded
dataset = dataset.drop(['Vehicle_IMD_Decile', 'Driver_IMD_Decile'], axis=1)

# Left or right hand drive is not useful since almost every value
# is no and the proportion of serious accidents is almost constant
dataset = dataset.drop(['Was_Vehicle_Left_Hand_Drive?'], axis=1)


# Almost every vehicle is not articulated
dataset['Towing_and_Articulation'] = (~dataset['Towing_and_Articulation'].isin(
    ['-1', 'No tow/articulation']
))

# Deleting this variable because almost every value is none
dataset = dataset.drop('Hit_Object_in_Carriageway', axis=1)

# Since we don't know the meaning of this variable and we cannot
# obtain it, we decide to drop this variable since it's not very informative
dataset = dataset.drop(
    'Driver_Home_Area_Type',
    axis=1
)

# This variable gets converted into 0-1 encoding whether
# the vehicle has abandoned the carriageway or not
dataset['Carriageway_Left'] = (dataset['Vehicle_Leaving_Carriageway'] != 'Did not leave carriageway')*1

dataset = dataset.drop('Vehicle_Leaving_Carriageway', axis=1)

# Deleting it since it is not very informative variable for the algorithm
dataset = dataset.drop('Journey_Purpose_of_Driver', axis=1)

# Junction_Location collides with another column from accidents. Deleting
dataset = dataset.drop('Junction_Location', axis=1)

dataset['Skidding_and_Overturning'] = dataset[
    dataset['Skidding_and_Overturning'].isin(['-1', 'None'])
]

# MISSING DATA HANDLING
# Missing values for categorical data are replaced with the mode of the variable (common point imputation)
dataset.loc[dataset['Vehicle_Type'] == '-1', 'Vehicle_Type'] = dataset['Vehicle_Type'].mode()

dataset.loc[
    dataset['Vehicle_Location-Restricted_Lane'] == '-1', 'Vehicle_Location-Restricted_Lane'
] = dataset['Vehicle_Location-Restricted_Lane'].mode()

dataset.loc[
    dataset['Hit_Object_off_Carriageway'] == '-1', 'Hit_Object_off_Carriageway'
] = dataset['Hit_Object_off_Carriageway'].mode()

# Obtention of remaining onehot variables
dataset = pd.get_dummies(dataset)
dataset = pd.get_dummies(
    dataset,
    columns=['road_type', 'weather_conditions']
)

# Deletion of columns relative to -1
dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='-1')))]

# Dataset permutation
dataset = dataset.sample(frac=1, random_state=42)

x = dataset.drop('target', axis=1)
y = dataset['target']

skf = model_selection.StratifiedKFold(n_splits=5)

classifier = ensemble.RandomForestClassifier(
    n_estimators=100, max_depth=20, class_weight='balanced'
)

split_indices = skf.split(x, y)
scores = []
f1_scores = []
confusion_matrices = []

for train_index, test_index in split_indices:
    print("New split")
    print("Train size: {}, test size: {}".format(
        len(train_index), len(test_index)
    ))

    train_set, train_labels = x.loc[train_index], y[train_index]
    test_set, test_labels = x.loc[test_index], y[test_index]

    classifier.fit(train_set, train_labels)

    labels_pred = classifier.predict(test_set)

    curr_score = metrics.accuracy_score(test_labels, labels_pred)
    curr_f1 = metrics.f1_score(test_labels, labels_pred, average='binary')
    conf_matrix = metrics.confusion_matrix(test_labels, labels_pred)
    print("Acc: {}, F1: {}".format(curr_score, curr_f1))
    print("Confusion matrix")
    print(conf_matrix)
    scores.append(curr_score)
    f1_scores.append(curr_f1)

print("Medias")
print("Acc: {}, F1: {}".format(np.mean(scores), np.mean(f1_scores)))
