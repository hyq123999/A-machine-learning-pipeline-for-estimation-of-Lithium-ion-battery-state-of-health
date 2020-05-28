import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob, os
import math
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import RandomizedLasso
from sklearn.linear_model import ElasticNet, Lasso, Ridge

# scaling
from sklearn.preprocessing import StandardScaler



'''
Pipeline for the custom test-split function
'''
def pipeline(root, path_c, split_percentage):
    '''REVISED CODE'''
    '''
    path_c = capacity data path
    path_c_v = capacity & voltage data path
    split_percentage = percentage of how to split the data
    '''

    data_c = pd.read_csv(root + path_c, index_col=0)  # capacity data


    # nominal battery capacity as per the datasheet
    Nominal_Capacity = data_c['Discharge_Q'][1]

    data = data_c.drop(['Discharge_Q'], axis=1)
    data_c_predict = data

    '''Split the data in into train and test'''
    split = math.ceil(split_percentage * len(data_c_predict)) # train on % of the number of cycles
    print('\nThe number of cycles in the data set {}: {} cycles\n'.format(path_c, len(data_c)))
    # print('The number of cycles the model has been trained on: {}\n'.format(split))

    # split the data
    train = data_c_predict[:split]
    test = data_c_predict[split:]

    train_x = train.drop(['SOH_discharge_capacity'], axis=1) # 'Cycle_Index_ordered'
    train_y = train['SOH_discharge_capacity']
    test_x = test.drop(['SOH_discharge_capacity'], axis=1) # 'Cycle_Index_ordered'
    test_y = test['SOH_discharge_capacity']

    '''Scale the datasets'''
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(train_x), columns=train_x.columns, index=train_x.index) #train_x #pd.DataFrame(scaler.fit_transform(train_x), columns=train_x.columns, index=train_x.index)
    y_train = train_y

    if split_percentage == 1:
        X_test = X_train
        y_test = y_train
    else:
        '''Scale data using the standard scaller'''
        X_test = pd.DataFrame(scaler.fit_transform(test_x), columns=test_x.columns, index=test_x.index) #test_x#pd.DataFrame(scaler.fit_transform(test_x), columns=test_x.columns, index=test_x.index)
        y_test = test_y

    return data_c, X_train, y_train, X_test, y_test, Nominal_Capacity
'''
Scoring function'''
def scoring_models(y_hat, y, estimator_name):
    # note estimator name must be string
    from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
    from sklearn.metrics import mean_squared_log_error, median_absolute_error, r2_score

    print('___Scores for the {} regression___'.format(estimator_name))
    print('|Explained variance:                     {}|'.format(explained_variance_score(y_hat, y, multioutput='raw_values')))
    print('|Mean absolute error:                    {}|'.format(mean_absolute_error(y_hat, y, multioutput='raw_values')))
    print('|Mean squared error:                     {}|'.format(mean_squared_error(y_hat, y, multioutput='raw_values')))
    print('|Mean squared logarithmic error:         {}|'.format(mean_squared_log_error(y_hat, y, multioutput='raw_values')))
    print('|Median absolute error:          {}|'.format(median_absolute_error(y_hat, y)))
    print('|R^2 (coefficient of determination):    {}|'.format(r2_score(y_hat, y, multioutput='raw_values')))
    print('______________________________________________________')

def plot_features_RFE(training_data, estimator_type, estimator_name):
    feat_labels = list(training_data)
    importances = estimator_type.feature_importances_

    indices = np.argsort(importances)[::-1]

    for f in range(training_data.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30,
                                feat_labels[indices[f]],
                                importances[indices[f]]))

    plt.figure(figsize=(20, 20))
    plt.title('Feature ranking for {}'.format(estimator_name), fontsize=26)
    plt.bar(range(training_data.shape[1]),
            importances[indices],
            color='lightblue',
            align='center')


# master root since the code is in a different folder
root_file = '/Users/dariusroman/Google Drive/Prognsotics/PhD Embedded Intelligence/Algorithm comparison on CALCE data/Python Code/'

# select relevant dataset
# dataset_folder = ['1C discharge_charge', '0.5C discharge_charge']
dataset_folder = ['FDOD Training Data']
# dataset = ['35', '36', '37', '38']
dataset = ['33_35', '34_35'] # dataset combinations
# dataset_type = ['data_historical_cumulative_CS2_', 'data_historical_1cycle_lag_CS2_', 'data_features_during_cycle_CS2_',
#                 'capacity_fade_data_CS2_']
dataset_type = ['data_historical_cumulative_CS2_', 'data_historical_1cycle_lag_CS2_', 'data_features_during_cycle_CS2_']

# determine the train and test data
# for i in dataset_folder:
#     for j in dataset:
i = dataset_folder[0] # '1C discharge_charge', '0.5C discharge_charge'
j = dataset[0] # '35', '36', '37', '38'
# k = dataset_type[0] # 'data_historical_cumulative_CS2_', 'data_historical_1cycle_lag_CS2_', 'data_features_during_cycle_CS2_'

for k in dataset_type:
    # select relevant path depending on analysis : 1). individual dataset pr 2). combinationwha
    # path_file = 'CALCE Data/Prismatic Cell/CS2/' + i + '/CS2_' + j + '/' + k +j+ '.csv'
    path_file = 'CALCE Data/Prismatic Cell/CS2/' + i + '/' + k +j+ '.csv'

    # generate the training data and fit the modela
    percentage = 1

    data_train, X_train, y_train, c,  d, Nominal_train = pipeline(root_file, path_file, percentage)

    # path_file_test = 'CALCE Data/Prismatic Cell/CS2/' + i + '/CS2_' + j_test + '/' + k + j_test + '.csv'
    path_file_test = 'CALCE Data/Prismatic Cell/CS2/' + '1C discharge_charge' + '/CS2_' + '37' + '/' + k + '37' + '.csv'
    data_test, a, b, X_test, y_test, Nominal_test = pipeline(root_file, path_file_test, percentage)


    # use the shrinkage method rank features
    '''Shrinkage method using the Lasso regressor'''
    '''Recursive Feature Elimination'''

    no_of_features = 1  #The number of features to select. If None, half of the features are selected.


    names = list(X_train)  # feature label

    print('Attributes are: {}'.format(list(X_train)))

    '''Recursive Feature Elimination'''

    '''Recursive Feature Elimination using SVR'''
    # '''
    estimator_SVR = SVR(kernel='linear', C=10, coef0=1, epsilon=0.05)
    selector = RFE(estimator_SVR, no_of_features, step=1)
    selector_SVR = selector.fit(X_train, y_train)
    print("Features sorted by their score using SVR:")
    svr_features = sorted(zip(map(lambda x: round(x, 4), selector_SVR.ranking_), names), reverse=False)
    print(svr_features)

    print('Accuracy scores based on offline training and online deployment (no-retraining)')
    predict_RFE_SVR = selector_SVR.predict(X_test)
    # scoring_models(predict_RFE_SVR, y_test, 'Random Forest')
    print('Score from the RFE function {}'.format(selector_SVR.score(X_test, y_test)))

    # '''
    '''Recursive Feature Elimination using RF'''

    estimator_RF = RandomForestRegressor(n_estimators=100, criterion='mse', min_samples_split=10, max_depth=7, min_samples_leaf=3)
    selector = RFE(estimator_RF, no_of_features, step=1)
    selector_RF = selector.fit(X_train, y_train)
    print("Features sorted by their score using RF:")
    rf_features = sorted(zip(map(lambda x: round(x, 4), selector_RF.ranking_), names), reverse=False)
    print(rf_features)

    print('Accuracy scores based on offline training and online deployment (no-retraining)')
    predict_RFE_RF = selector_RF.predict(X_test)
    # scoring_models(predict_RFE_RF, y_test, 'Random Forest')
    print('Score from the RFE function {}'.format(selector_RF.score(X_test, y_test)))


