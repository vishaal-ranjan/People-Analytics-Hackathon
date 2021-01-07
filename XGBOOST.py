import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

# load data
data = pd.read_excel('Health_Care_Data_train.xlsx')

# Converted ‘Specialty’ to 1-Hot, dropped “Group”
dummy = pd.get_dummies(data['Specialty'])
df = pd.concat([dummy, data], axis = 1)
df.drop(['Specialty', 'Group'], axis = 1, inplace=True)

# Drooped rows with NAN value in column “q58”
df = df.dropna(subset=['q58'])
df.reset_index(drop=True, inplace=True)

# Replaced missing cells with the most frequent value of part of their column who have the same value of “q58”
fre = df.groupby('q58',as_index=False).agg(lambda x:x.value_counts().index[0])
for col in df:
    if col != 'q58':
        for i, row_value in df[col].iteritems():
            if np.isnan(row_value):
                df[col][i] = fre.loc[fre['q58'] == df['q58'][i]][col].values[0]
dataset = np.array(df)

# separate dataset to train dataset and test dataset
X = dataset[:, 0:40].astype('int')
Y = dataset[:, 40].astype('int')
seed = 10
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# run model with tuned parameters
model = xgb.XGBClassifier(max_depth=8,
                        learning_rate=0.01,
                        n_estimators=30,
                        silent=True,
                        objective='binary:logistic',
                        nthread=-1,
                        gamma=0,
                        min_child_weight=4,
                        max_delta_step=0,
                        subsample=0.85,
                        colsample_bytree=0.7,
                        colsample_bylevel=1,
                        reg_alpha=0,
                        reg_lambda=1,
                        scale_pos_weight=1,
                        seed=1440,
                        missing=None)
model.fit(X_train, y_train)

# select parameters with importance more than 0.00277
selection = SelectFromModel(model, threshold=0.00277, prefit=True)
select_X_train = selection.transform(X_train)
# train model
selection_model = xgb.XGBClassifier(max_depth=8,
                    learning_rate=0.01,
                    n_estimators=30,
                    silent=True,
                    objective='binary:logistic',
                    nthread=-1,
                    gamma=0,
                    min_child_weight=4,
                    max_delta_step=0,
                    subsample=0.85,
                    colsample_bytree=0.7,
                    colsample_bylevel=1,
                    reg_alpha=0,
                    reg_lambda=1,
                    scale_pos_weight=1,
                    seed=1440,
                    missing=None)
selection_model.fit(select_X_train, y_train)
# eval model
select_X_test = selection.transform(X_test)
y_pred = selection_model.predict(select_X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("n=%d, Accuracy: %.3f%%" % (select_X_train.shape[1], accuracy*100.0))