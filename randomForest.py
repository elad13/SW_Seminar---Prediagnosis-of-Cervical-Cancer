# import of libraries used
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics._classification import _check_set_wise_labels, multilabel_confusion_matrix, _prf_divide, _warn_prf

sns.set_style('darkgrid')

from scipy import stats
from scipy.stats import norm

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


from imblearn.over_sampling import SMOTE

from inspect import signature

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO

from IPython.display import display, Image

import pydotplus
#import xgboost
import shape


# load the dataset as a Pandas dataframe
# name of CSV file, na_values = string to recognize as NULL
df_import = pd.read_csv('kag_risk_factors_cervical_cancer.csv', na_values="?")

# look at the dataset overview
df_import.info()

# # look at the count of unique values in each factor, including NaN's
# for column in df_import.columns:
#     print('Unique values and count in {}'.format(column))
#     print(df_import[column].value_counts(dropna=False))

# view just the count of NaN's in each factor
df_import.isna().sum()

# copy the imported file to a new dataframe to retain the original imported file as a seperate object
df = df_import.copy(deep=True)


# function to create a new boolean column in a df
# parameters are a dataframe and a column name
# if any value was present for a record in the column provided, the output value will be "1",
# if the value tests true for pd.isna() the output value will be "0".
# returns a list of values corresponding to each record in the provided dataframe
def new_bool(df, col_name):
    bool_list = []
    for index, row in df.iterrows():
        #         print(row)
        value = row[col_name]
        #         print(value)
        value_out = 1
        if pd.isna(value):
            value_out = 0

        #       for testing
        #         print("value: {}   -   bool: {}".format(value, str(value_out)))

        bool_list.append(value_out)

    return bool_list

# create new factor 'is_number_partners_known'
df['is_number_partners_known'] = new_bool(df, 'Number of sexual partners')
# check if operation was successful
df['is_number_partners_known'].value_counts(dropna=False)

# create new factor 'is_first_intercourse_known'
df['is_first_intercourse_known'] = new_bool(df, 'First sexual intercourse')
# check if operation was successful
df['is_first_intercourse_known'].value_counts(dropna=False)

# create new factor 'is_number_pregnancies_known'
df['is_number_pregnancies_known'] = new_bool(df, 'Num of pregnancies')
# check if operation was successful
df['is_number_pregnancies_known'].value_counts(dropna=False)

df2 = df.copy(deep=True)

df2['STDs: Number of diagnosis'].value_counts(dropna=False)
# subset the records where the value is zero for 'STDs: Number of diagnosis'
value_zero = df2['STDs: Number of diagnosis'] == 0
temp = df2[value_zero]

#print('time since first: ', temp['STDs: Time since first diagnosis'].value_counts(dropna=False))
#print('time since last: ', temp['STDs: Time since last diagnosis'].value_counts(dropna=False))

# replace null value in 0 value
df2['STDs: Time since first diagnosis'].fillna(0, inplace=True)
df2['STDs: Time since first diagnosis'].value_counts(dropna=False)

df2['STDs: Time since last diagnosis'].fillna(0, inplace=True)
df2['STDs: Time since last diagnosis'].value_counts(dropna=False)


# function to display a countplot, boxplot, and summary stats for a factor
def countplot_boxplot(column, dataframe):
    # fig = plt.figure(figsize=(15, 20))
    # fig.suptitle(column, size=20)
    #
    # ax1 = fig.add_subplot(2, 1, 1)
    # sns.countplot(dataframe[column])
    # #     plt.title(column)
    # plt.xticks(rotation=45)
    #
    # ax2 = fig.add_subplot(2, 1, 2)
    # sns.boxplot(dataframe[column])
    # #     plt.title(column)
    # plt.xticks(rotation=45)
    # plt.show()

    print(column+':')
    print('Min:', dataframe[column].min())
    print('Mean:', dataframe[column].mean())
    print('Median:', dataframe[column].median())
    print('Mode:', dataframe[column].mode()[0])
    print('Max:', dataframe[column].max())
    print('**********************')
    print('% of values missing:', (df2[column].isna().sum() / len(df2)) * 100)


# function to replace missing values with the median
def fillna_median(column, dataframe):
    dataframe[column].fillna(dataframe[column].median(), inplace=True)
    print(dataframe[column].value_counts(dropna=False))


# function to replace missing values with the mean
def fillna_mean(column, dataframe):
    dataframe[column].fillna(dataframe[column].mean(), inplace=True)
    print(dataframe[column].value_counts(dropna=False))


# function to replace missing values with a value provided
def fillna_w_value(value, column, dataframe):
    dataframe[column].fillna(value, inplace=True)
    print(dataframe[column].value_counts(dropna=False))


col_list = ['Smokes', 'Smokes (years)', 'Smokes (packs/year)']
for col in col_list:
    fillna_w_value(0, col, df2)

col_list = ['Hormonal Contraceptives', 'IUD', 'STDs']
for col in col_list:
    print(col,':')
    print(df2[col].value_counts(dropna=False))
    print("----------------")

# replace more missing values and confirm results
fillna_w_value(1, 'Hormonal Contraceptives', df2)
fillna_w_value(0, 'IUD', df2)
fillna_w_value(0, 'STDs', df2)

countplot_boxplot('Number of sexual partners', df2)
# replace with median
fillna_median('Number of sexual partners', df2)

countplot_boxplot('First sexual intercourse', df2)
# replace with median
fillna_median('First sexual intercourse', df2)

countplot_boxplot('Num of pregnancies', df2)
# replace with median
fillna_median('Num of pregnancies', df2)

countplot_boxplot('Hormonal Contraceptives (years)', df2)
# replace with zeros
fillna_w_value(0,'Hormonal Contraceptives (years)', df2)

countplot_boxplot('IUD (years)', df2)
# replace with zeros
fillna_w_value(0, 'IUD (years)', df2)

countplot_boxplot('STDs (number)', df2)
# replace with zeros
fillna_w_value(0, 'STDs (number)', df2)

# replace missing values with a zero
col_list = [
    'STDs:condylomatosis',
    'STDs:cervical condylomatosis',
    'STDs:vaginal condylomatosis',
    'STDs:vulvo-perineal condylomatosis',
    'STDs:syphilis',
    'STDs:pelvic inflammatory disease',
    'STDs:genital herpes',
    'STDs:molluscum contagiosum',
    'STDs:AIDS',
    'STDs:HIV',
    'STDs:Hepatitis B',
    'STDs:HPV']
for col in col_list:
    fillna_w_value(0, col, df2)

# drop useless factors
#df2.drop(['STDs:cervical condylomatosis','STDs:AIDS'], axis=1, inplace=True)
df2.drop(['Smokes','Hormonal Contraceptives', 'IUD', 'STDs'], axis=1, inplace=True)

# look for any remaining NaN's in the dataframe
df2.isna().sum()

# Exploratory Data Analysis:

# # pull the target variable from the dataframe and use a heatmap to look at possible correlations between factors
# temp_df = df2.drop('Biopsy', axis=1)
# f,ax = plt.subplots(figsize=(20,20))
# sns.heatmap(temp_df.loc[:,:].corr(), annot=True, cmap="Blues", fmt='.1f' )
# plt.show()

# # look at a correlation matrix of the top 12 factors to the target variable: Biopsy
# corrmat = df2.corr()
# k = 12 #number of variables for heatmap
# cols = corrmat.nlargest(k, 'Biopsy')['Biopsy'].index
# cm = np.corrcoef(df2[cols].values.T)
#
# plt.figure(figsize=(15,15))
#
# sns.set(font_scale=2)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
#                  yticklabels = cols.values, xticklabels = cols.values)
# plt.show()

# make a backup copy of df2 at this stage of processing
df_backup = df2.copy()

#Training and Testing Machine Learning Models:

# create X and y
X = df2.drop(['Biopsy'], axis=1)
y = df2['Biopsy']
#X = df2.drop(['Hinselmann','Schiller','Citology','Biopsy'], axis=1)
#y = df2[['Hinselmann','Schiller','Citology','Biopsy']]

X.info()
y.head()
#y.info()

# create an 80/20 train/test split with a specified random value for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state= 10)

# standardize the continuous factors
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))

X_train = pd.DataFrame(minmax_scale.fit_transform(X_train), columns = X.columns)
X_test = pd.DataFrame(minmax_scale.fit_transform(X_test), columns = X.columns)

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)

y_train.value_counts()
y_test.value_counts()

#not implemented
#Creating an additional test/train dataset with "synthetic" training samples

#Machine Learning Models:
# create a function to display a graph of precision and recall and the scores I am interested in
def analysis(model, X_train, y_train):
    model.fit(X_train, y_train)

    # predict probabilities
    probs = model.predict_proba(X_test)
    print("###############")
    print("probs: ", probs)
    print("###############")
    # keep probabilities for the positive outcome only
    probs = probs[:, 1]

    # predict class values
    preds = model.predict(X_test)

    # calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, probs)

    # calculate average precision
    average_precision = average_precision_score(y_test, probs)

    # recall score for class 1 (Predict that Biopsy is True)
    rs = recall_score(y_test, preds)

    # calculate F1 score
    # precision_recall_fscore_support return 4 things, I need just the f
    sample_weight = None
    warn_for = ('f-score')
    zero_division = "warn"
    average = 'binary'
    beta = 1.0
    #_check_zero_division("warn")
    labels = _check_set_wise_labels(y_test, preds, average, None, 1)

    # Calculate tp_sum, pred_sum, true_sum ###
    samplewise = average == 'samples'
    MCM = multilabel_confusion_matrix(y_test, preds, sample_weight=None, labels=labels, samplewise=samplewise)

# In multilabel confusion matrix :math:`MCM`, the count of true negatives
#     is :math:`MCM_{:,0,0}`, false negatives is :math:`MCM_{:,1,0}`,
#     true positives is :math:`MCM_{:,1,1}` and false positives is
#     :math:`MCM_{:,0,1}`.
    tn_sum = MCM[:, 0, 0]
    fn_sum = MCM[:, 1, 0]
    fp_sum = MCM[:, 0, 1]

    tp_sum = MCM[:, 1, 1]
    pred_sum = tp_sum + MCM[:, 0, 1]
    true_sum = tp_sum + MCM[:, 1, 0]

    if average == 'micro':
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])
        # added
        tn_sum = np.array([tn_sum.sum()])
        fn_sum = np.array([fn_sum.sum()])
        fp_sum = np.array([fp_sum.sum()])

    # Finally, we have all our sufficient statistics. Divide! #
    beta2 = beta ** 2

    # Divide, and on zero-division, set scores and/or warn according to
    # zero_division:
    precision1 = _prf_divide(tp_sum, pred_sum, 'precision',
                            'predicted', average, warn_for, zero_division)
    recall1 = _prf_divide(tp_sum, true_sum, 'recall',
                            'true', average, warn_for, zero_division)

    # warn for f-score only if zero_division is warn, it is in warn_for
    # and BOTH prec and rec are ill-defined
    if zero_division == "warn" and ("f-score",) == warn_for:
        if (pred_sum[true_sum == 0] == 0).any():
            _warn_prf(
                average, "true nor predicted", 'F-score is', len(true_sum)
            )

    # if tp == 0 F will be 1 only if all predictions are zero, all labels are
    # zero, and zero_division=1. In all other case, 0
    if np.isposinf(beta):
        f_score = recall1
    else:
        denom = beta2 * precision1 + recall1

        denom[denom == 0.] = 1  # avoid division by 0
        f_score = (1 + beta2) * precision1 * recall1 / denom

    # Average the results
    if average == 'weighted':
        weights = true_sum
        if weights.sum() == 0:
            zero_division_value = 0.0 if zero_division in ["warn", 0] else 1.0
            # precision is zero_division if there are no positive predictions
            # recall is zero_division if there are no positive labels
            # fscore is zero_division if all labels AND predictions are
            # negative
            return (zero_division_value if pred_sum.sum() == 0 else 0,
                    zero_division_value,
                    zero_division_value if pred_sum.sum() == 0 else 0,
                    None)

    elif average == 'samples':
        weights = sample_weight
    else:
        weights = None

    if average is not None:
        assert average != 'binary' or len(precision1) == 1
        precision1 = np.average(precision1, weights=weights)
        recall1 = np.average(recall1, weights=weights)
        f_score = np.average(f_score, weights=weights)
        true_sum = None  # return no support

    #return precision, recall, f_score, true_sum
    print("precision: ", precision1)
    print("recall: ", recall1)
    print("tp_sum: ", tp_sum)
    print("tn_sum: ", tn_sum)
    print("fn_sum: ", fn_sum)
    print("fp_sum: ", fp_sum)

    #f1 = f1_score(y_test, preds)
    f1 = f_score

    # calculate precision-recall AUC
    auc_score = auc(recall, precision)

    # create chart
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    # plot a "no skill" line
    plt.plot([0, 1], [0.5, 0.5], linestyle='--')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: Average Precision={0:0.3f}'.format(average_precision))
    plt.show()

    # print(confusion_matrix(y_test, preds))
    print('Classification Report:')
    print(classification_report(y_test, preds))

    print('f1=%.3f auc=%.3f recall=%.3f' % (f1, auc_score, rs))

# function to plot the importances of factors used in a model
def plot_feature_importances(model):
    n_features = X_train.shape[1]
    plt.figure(figsize=(15,15))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")



# creating the model
model = RandomForestClassifier()

# feeding the training data into the model
for i in range(10):
    model.fit(X_train, y_train)

# predicting the test set results
y_pred = model.predict(X_test)

# Calculating the accuracies
print("Training accuracy :", model.score(X_train, y_train))
print("Testing accuracy :", model.score(X_test, y_test))

# classification report
print(classification_report(y_test, y_pred))

# confusion matrix: TP, TN, FP, FN
print(confusion_matrix(y_test, y_pred))

#Random Forest
# use grid search for the random forest classifier, use recall as the factor to optimize
# forest = RandomForestClassifier(random_state=10, n_jobs=-1)
#
# forest_param_grid = {
#     'class_weight': ['balanced'],
#     'criterion': ['gini', 'entropy' ],
#     'max_depth': [2, 3, 4, 5, 6, 7, 8],
#     'n_estimators': [20, 40, 50, 60, 80, 100, 200]} #number of trees in the foreset
#
# forest_grid_search = GridSearchCV(forest,
#                                   param_grid = forest_param_grid,
#                                   scoring = 'recall',
#                                   cv=10,
#                                   return_train_score=True)
#
# import time
# start = time.time()
#
# forest_grid_search.fit(X_train, y_train)
#
# print("Testing Accuracy: {:.4}%".format(forest_grid_search.best_score_ * 100))
# print("Total Runtime for Grid Search on Random Forest Classifier: {:.4} seconds".format(time.time() - start))
# print("")
# print("Optimal Parameters: {}".format(forest_grid_search.best_params_))
#
# forest_param_grid = {
#     'class_weight': ['balanced'],
#     'criterion': ['gini'],
#     'max_depth': [2, 3, 4],
#     'n_estimators': [10, 15, 20, 25, 30]}
#
# forest_grid_search = GridSearchCV(forest,
#                                   param_grid = forest_param_grid,
#                                   scoring = 'recall',
#                                   cv=10,
#                                   return_train_score=True)
#
# import time
# start = time.time()
#
# forest_grid_search.fit(X_train, y_train)
#
# print("Testing Accuracy: {:.4}%".format(forest_grid_search.best_score_ * 100))
# print("Total Runtime for Grid Search on Random Forest Classifier: {:.4} seconds".format(time.time() - start))
# print("")
# print("Optimal Parameters: {}".format(forest_grid_search.best_params_))
#
# forest = RandomForestClassifier(n_estimators=10,
#                                 criterion='gini',
#                                 max_depth=2,
#                                 class_weight='balanced',
#                                 random_state=10)
# analysis(forest, X_train, y_train)
#
# plot_feature_importances(forest)
# plt.show()
#
# # Plot the feature importances of the forest
# plt.figure(figsize=(20,20))
# plt.title("Feature importances")
#
# importances = forest.feature_importances_
#
# std = np.std([tree.feature_importances_ for tree in forest.estimators_],
#              axis=0)
#
# labels = np.array(X.columns)
#
# label_importance_std = pd.DataFrame(columns=['Factor','Importance', 'STD'])
# label_importance_std['Factor'] = labels
# label_importance_std['Importance'] = importances
# label_importance_std['STD'] = std
#
# label_importance_std.sort_values('Importance', inplace=True, ascending=False)
#
#
# #make the graph
# plt.bar(range(X_train.shape[1]), label_importance_std['Importance'],
#        color="r", yerr=label_importance_std['STD'], align="center")
# plt.xticks(range(X_train.shape[1]), label_importance_std['Factor'], rotation=90)
# plt.xlim([-1, X_train.shape[1]])
# plt.show()


# # use grid search for the random forest classifier, use recall as the factor to optimize - resampled, balanced data
# forest = RandomForestClassifier(random_state=10, n_jobs=-1)
#
# forest_param_grid = {
#     'class_weight': [None],
#     'criterion': ['gini', 'entropy'],
#     'max_depth': [5, 6, 7, 8, 9, 10, 11, 12],
#     'n_estimators': [10, 20, 30, 40, 50, 60]}
#
# forest_grid_search = GridSearchCV(forest,
#                                   param_grid = forest_param_grid,
#                                   scoring = 'recall',
#                                   cv=3,
#                                   return_train_score=True)
#
# import time
# start = time.time()
#
# forest_grid_search.fit(X_train_res, y_train_res)
#
# print("Testing Accuracy: {:.4}%".format(forest_grid_search.best_score_ * 100))
# print("Total Runtime for Grid Search on Random Forest Classifier: {:.4} seconds".format(time.time() - start))
# print("")
# print("Optimal Parameters: {}".format(forest_grid_search.best_params_))
#
# # refine the gridsearch based on the results of this test
# forest = RandomForestClassifier(random_state=10, n_jobs=-1)
#
# forest_param_grid = {
#     'class_weight': [None],
#     'criterion': ['gini'],
#     'max_depth': [7,8,9],
#     'n_estimators': [35, 40, 45]}
#
# forest_grid_search = GridSearchCV(forest,
#                                   param_grid = forest_param_grid,
#                                   scoring = 'recall',
#                                   cv=3,
#                                   return_train_score=True)
#
# import time
# start = time.time()
#
# forest_grid_search.fit(X_train_res, y_train_res)
#
# print("Testing Accuracy: {:.4}%".format(forest_grid_search.best_score_ * 100))
# print("Total Runtime for Grid Search on Random Forest Classifier: {:.4} seconds".format(time.time() - start))
# print("")
# print("Optimal Parameters: {}".format(forest_grid_search.best_params_))
#
# #create the model using the best parameters
# forest = RandomForestClassifier(criterion='gini',
#                                              max_depth=8,
#                                              n_estimators=40,
#                                              random_state=10,
#                                              n_jobs=-1)
#
# analysis(forest, X_train_res, y_train_res)