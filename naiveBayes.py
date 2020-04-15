# import of libraries used
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB, MultinomialNB

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
    f1 = f1_score(y_test, preds)

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


# Naive Bayes
# https://github.com/angelinap/Cervical-Cancer-Risk-Factors/blob/master/finalproject.py
print ("-----Gaussian Naive Bayes-----")
GaussNB = GaussianNB()
# GaussNB.fit(x_train, y_train)
GaussNB.fit(X_train, y_train)
myprediction4 = GaussNB.predict(X_test)

print (np.shape(myprediction4), np.shape(y_test))
print ("Gaussian Naive Bayes Accuracy...")
# score = GaussNB.score(x_test, y_test)
score = accuracy_score(y_test, myprediction4)
print (score)

# https://github.com/arunravishankar/Cervical_cancer_risk_classification/blob/master/Cervical_cancer.ipynb
# def model_efficacy(conf):
#     total_num = np.sum(conf)
#     sen = conf[0][0] / (conf[0][0] + conf[1][0])
#     spe = conf[1][1] / (conf[1][0] + conf[1][1])
#     false_positive_rate = conf[0][1] / (conf[0][1] + conf[1][1])
#     false_negative_rate = conf[1][0] / (conf[0][0] + conf[1][0])
#
#     print('total_num: ', total_num)
#     print('G1P1: ', conf[0][0])  # G = gold standard; P = prediction
#     print('G0P1: ', conf[0][1])
#     print('G1P0: ', conf[1][0])
#     print('G0P0: ', conf[1][1])
#     print('##########################')
#     print('sensitivity: ', sen)
#     print('specificity: ', spe)
#     print('false_positive_rate: ', false_positive_rate)
#     print('false_negative_rate: ', false_negative_rate)
#
#     return total_num, sen, spe, false_positive_rate, false_negative_rate
#
#
# nb=GaussianNB()
# nb.fit(X_train, y_train) #df_train_feature, train_label)
# predictionbayes=nb.predict(X_test)
# scores = nb.score(X_test, y_test)
# print('accuracy=',scores)
#
# df_ansbayes = pd.DataFrame({'Biopsy' :y_test})
# df_ansbayes['Prediction'] = predictionbayes
# df_ansbayes[ df_ansbayes['Biopsy'] != df_ansbayes['Prediction'] ]
#
# cols = ['Biopsy_1','Biopsy_0']  #Gold standard
# rows = ['Prediction_1','Prediction_0'] #diagnostic tool (our prediction)
#
# B1P1bayes = len(df_ansbayes[(df_ansbayes['Prediction'] == df_ansbayes['Biopsy']) & (df_ansbayes['Biopsy'] == 1)])
# B1P0bayes = len(df_ansbayes[(df_ansbayes['Prediction'] != df_ansbayes['Biopsy']) & (df_ansbayes['Biopsy'] == 1)])
# B0P1bayes = len(df_ansbayes[(df_ansbayes['Prediction'] != df_ansbayes['Biopsy']) & (df_ansbayes['Biopsy'] == 0)])
# B0P0bayes = len(df_ansbayes[(df_ansbayes['Prediction'] == df_ansbayes['Biopsy']) & (df_ansbayes['Biopsy'] == 0)])
#
# confbayes = np.array([[B1P1bayes,B0P1bayes],[B1P0bayes,B0P0bayes]])
# df_cmbayes = pd.DataFrame(confbayes, columns = [i for i in cols], index = [i for i in rows])
#
# f, ax= plt.subplots(figsize = (5, 5))
# sns.heatmap(df_cmbayes, annot=True, ax=ax)
# ax.xaxis.set_ticks_position('top') #Making x label be on top is common in textbooks.
#
# print('total test case number: ', np.sum(confbayes))
# model_efficacy(confbayes)




