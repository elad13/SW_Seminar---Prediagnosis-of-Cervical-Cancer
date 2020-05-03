import pandas as pd
import preprocessing as preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', 1500)
pd.set_option('display.max_columns', 20000)
pd.set_option('display.width', 1000)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

df_full=pd.read_csv('kag_risk_factors_cervical_cancer.csv', na_values="?")
#df_full.shape

#df_full['Biopsy']
df_full.info()

df_full.isna().sum()
#df_fullna = df_full.replace('?', np.nan)
#df_fullna.isnull().sum()

df = df_full.copy(deep=True)
#df=df_fullna
#df.columns
# df = df.apply(pd.to_numeric, axis=0)
# df.info()
# df.describe()
# df=df.drop(['Smokes','Hormonal Contraceptives', 'IUD', 'STDs'], axis=1, inplace=True)
# cols = list(df.columns)
# mediancolumns = ['Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies',
#                  'Smokes (years)', 'Smokes (packs/year)', 'Hormonal Contraceptives (years)', 'STDs (number)',
#                  'STDs:condylomatosis', 'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis',
#                  'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis', 'STDs:pelvic inflammatory disease',
#                  'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV', 'STDs:Hepatitis B',
#                  'STDs:HPV']
# for i, colm in enumerate(mediancolumns):
#     df[colm] = df[colm].fillna(df[colm].median())

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


# categoricalcolumns = ['Smokes', 'Hormonal Contraceptives', 'IUD', 'IUD (years)', 'STDs']
# for i, colm in enumerate(categoricalcolumns):
#     df[colm] = df[colm].fillna(1)
#
# # for categorical variable I need to understand what this does!
# df = pd.get_dummies(data=df, columns=['Smokes', 'Hormonal Contraceptives', 'IUD', 'STDs',
#                                       'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx', 'Hinselmann', 'Citology', 'Schiller'])
# df.isnull().sum()
# df_data['Biopsy'].sum()
# df_data=df # Saving data
# df.describe(include='all')
# fig, (ax1,ax2,ax3,ax4,ax5,ax6,ax7) = plt.subplots(7,1,figsize=(20,40))
# sns.countplot(x='Age', data=df, ax=ax1)
# sns.countplot(x='Number of sexual partners', data=df, ax=ax2)
# sns.countplot(x='Num of pregnancies', data=df, ax=ax3)
# sns.countplot(x='Smokes (years)', data=df, ax=ax4)
# sns.countplot(x='Hormonal Contraceptives (years)', data=df, ax=ax5)
# sns.countplot(x='IUD (years)', data=df, ax=ax6)
# sns.countplot(x='STDs (number)', data=df, ax=ax7)

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



# sns.jointplot(x='Age', y='Biopsy', data=df, alpha=0.1) #Alpha let's you see the points
#
# fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(15,12))
# sns.countplot(x='Age', data=df, ax=ax1)
# sns.countplot(x='Biopsy', data=df, ax=ax2)
# sns.barplot(x='Age', y='Biopsy', data=df, ax=ax3)
#
# #Stratified
# facet = sns.FacetGrid(df, hue='Biopsy',aspect=4)
# facet.map(sns.kdeplot,'Age',shade= True)
# facet.set(xlim=(0, df['Age'].max()))
# facet.add_legend()
#
# sns.jointplot(x='Number of sexual partners', y='Biopsy', data=df, alpha=0.1)
#
# fig, (ax1,ax2) = plt.subplots(2,1,figsize=(15,8))
# sns.countplot(x='Number of sexual partners', data=df, ax=ax1)
# sns.barplot(x='Number of sexual partners', y='Biopsy', data=df, ax=ax2) #categorical to categorical
#
# #continuous to categorical
# facet = sns.FacetGrid(df, hue='Biopsy',aspect=4)
# facet.map(sns.kdeplot,'Number of sexual partners',shade= True)
# facet.set(xlim=(0, df['Number of sexual partners'].max()))
# facet.add_legend()
#
# sns.jointplot(x='Num of pregnancies', y='Biopsy', data=df, alpha=0.1)
# sns.factorplot('Num of pregnancies','Biopsy',data=df, size=5, aspect=3)
#
# #continuous to categorical
# facet = sns.FacetGrid(df, hue='Biopsy',aspect=4)
# facet.map(sns.kdeplot,'Num of pregnancies',shade= True)
# facet.set(xlim=(0, df['Num of pregnancies'].max()))
# facet.add_legend()
#
# corrmat = df.corr()
# f, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=1, square=True, cmap='rainbow')
#
# k = 15 #number of variables for heatmap
# cols = corrmat.nlargest(k, 'Biopsy')['Biopsy'].index
# cm = np.corrcoef(df[cols].values.T)
#
# plt.figure(figsize=(9,9))
#
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
#                  yticklabels = cols.values, xticklabels = cols.values)
# plt.show()
#
# np.random.seed(24)
# df_data_shuffle = df_data.iloc[np.random.permutation(len(df_data))]
#
# df_train = df_data_shuffle.iloc[1:686, :] #80 percent of the data
# df_test = df_data_shuffle.iloc[686: , :]
#
# features=['Age', 'Number of sexual partners', 'First sexual intercourse',
#        'Num of pregnancies', 'Smokes (years)', 'Smokes (packs/year)',
#        'Hormonal Contraceptives (years)', 'IUD (years)', 'STDs (number)',
#        'STDs:condylomatosis', 'STDs:cervical condylomatosis',
#        'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis',
#        'STDs:syphilis', 'STDs:pelvic inflammatory disease',
#        'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS',
#        'STDs:HIV', 'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis',
#        'Smokes_0.0', 'Smokes_1.0',
#        'Hormonal Contraceptives_0.0', 'Hormonal Contraceptives_1.0', 'IUD_0.0',
#        'IUD_1.0', 'STDs_0.0', 'STDs_1.0', 'Dx:Cancer_0', 'Dx:Cancer_1',
#        'Dx:CIN_0', 'Dx:CIN_1', 'Dx:HPV_0', 'Dx:HPV_1', 'Dx_0', 'Dx_1',
#        'Hinselmann_0', 'Hinselmann_1', 'Citology_0', 'Citology_1','Schiller_0','Schiller_1']
# len(features)

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

def analysis2(model, X_train, y_train):
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



# df_train.shape
# df_train_feature = df_train[features]
# train_label = np.array(df_train['Biopsy'])
# df_test_feature=df_test[features]
# test_label=np.array(df_test['Biopsy'])
#
# #Normalization
# from sklearn import preprocessing
# minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
# train_feature = minmax_scale.fit_transform(df_train_feature)
# test_feature = minmax_scale.fit_transform(df_test_feature)
#
# #Make sure if it's the shape what we want! And it is.
# print(train_feature[0])
# print(train_label[0])
# print(test_feature[0])
# print(test_label[0])
#
# train_feature.shape


# Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(X_train, y_train)
predictionbayes=nb.predict(X_test)
scores = accuracy_score(y_test, predictionbayes)
#scores = nb.score(X_test, test_label)
print('accuracy=',scores)

df_ansbayes = pd.DataFrame({'Biopsy' :predictionbayes})
df_ansbayes['Prediction'] = predictionbayes
df_ansbayes[ df_ansbayes['Biopsy'] != df_ansbayes['Prediction'] ]

cols = ['Biopsy_1','Biopsy_0']  #Gold standard
rows = ['Prediction_1','Prediction_0'] #diagnostic tool (our prediction)

B1P1bayes = len(df_ansbayes[(df_ansbayes['Prediction'] == df_ansbayes['Biopsy']) & (df_ansbayes['Biopsy'] == 1)])
B1P0bayes = len(df_ansbayes[(df_ansbayes['Prediction'] != df_ansbayes['Biopsy']) & (df_ansbayes['Biopsy'] == 1)])
B0P1bayes = len(df_ansbayes[(df_ansbayes['Prediction'] != df_ansbayes['Biopsy']) & (df_ansbayes['Biopsy'] == 0)])
B0P0bayes = len(df_ansbayes[(df_ansbayes['Prediction'] == df_ansbayes['Biopsy']) & (df_ansbayes['Biopsy'] == 0)])

confbayes = np.array([[B1P1bayes,B0P1bayes],[B1P0bayes,B0P0bayes]])
df_cmbayes = pd.DataFrame(confbayes, columns = [i for i in cols], index = [i for i in rows])

f, ax= plt.subplots(figsize = (5, 5))
sns.heatmap(df_cmbayes, annot=True, ax=ax)
ax.xaxis.set_ticks_position('top') #Making x label be on top is common in textbooks.

print('total test case number: ', np.sum(confbayes))


def model_efficacy(conf):
    total_num = np.sum(conf)
    sen = conf[0][0] / (conf[0][0] + conf[1][0])
    spe = conf[1][1] / (conf[1][0] + conf[1][1])
    false_positive_rate = conf[0][1] / (conf[0][1] + conf[1][1])
    false_negative_rate = conf[1][0] / (conf[0][0] + conf[1][0])

    print('total_num: ', total_num)
    print('G1P1: ', conf[0][0])  # G = gold standard; P = prediction
    print('G0P1: ', conf[0][1])
    print('G1P0: ', conf[1][0])
    print('G0P0: ', conf[1][1])
    print('##########################')
    print('sensitivity: ', sen)
    print('specificity: ', spe)
    print('false_positive_rate: ', false_positive_rate)
    print('false_negative_rate: ', false_negative_rate)

    return total_num, sen, spe, false_positive_rate, false_negative_rate

model_efficacy(confbayes)
