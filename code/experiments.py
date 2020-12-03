import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
import evaluation
from sklearn.utils import shuffle

random_state = 12
cutoff = 150


def linear_reg_all_data_all_features():
    data = pd.read_csv('../data/all_data.csv', index_col=0)
    experiment_name = linear_reg_all_data_all_features.__name__
    features = ['C', 'H', 'N', 'O', 'C Chain']
    X = data[features].values
    y = data[['extinction_coefficient']].values / 1000
    scalar = StandardScaler()
    X = scalar.fit_transform(X, y)
    X, y = shuffle(X, y, random_state=random_state)
    X_development, X_validate, y_development, y_validate = train_test_split(X, y, test_size=0.33,
                                                                            random_state=random_state, shuffle=True)
    model = LinearRegression()
    model.fit(X_development, y_development)
    result = evaluation.evaluate_regression_model(experiment_name, X_validate, y_validate, model)
    return result


def linear_reg_all_data_basic_features():
    data = pd.read_csv('../data/all_data.csv', index_col=0)
    experiment_name = linear_reg_all_data_basic_features.__name__
    features = ['C', 'H', 'N', 'O']
    X = data[features].values
    y = data[['extinction_coefficient']].values / 1000
    scalar = StandardScaler()
    X = scalar.fit_transform(X, y)
    X, y = shuffle(X, y, random_state=random_state)
    X_development, X_validate, y_development, y_validate = train_test_split(X, y, test_size=0.33,
                                                                            random_state=random_state, shuffle=True)
    model = LinearRegression()
    model.fit(X_development, y_development)
    result = evaluation.evaluate_regression_model(experiment_name, X_validate, y_validate, model)
    return result

def linear_reg_all_data_essential_features():
    data = pd.read_csv('../data/all_data.csv', index_col=0)
    experiment_name = linear_reg_all_data_essential_features.__name__
    X = data[['C', 'N']].values
    y = data[['extinction_coefficient']].values / 1000
    scalar = StandardScaler()
    X = scalar.fit_transform(X, y)
    X, y = shuffle(X, y, random_state=random_state)
    X_development, X_validate, y_development, y_validate = train_test_split(X, y, test_size=0.33,
                                                                            random_state=random_state, shuffle=True)
    model = LinearRegression()
    model.fit(X_development, y_development)
    result = evaluation.evaluate_regression_model(experiment_name, X_validate, y_validate, model)
    return result

def linear_reg_all_data_baseline_negative():
    data = pd.read_csv('../data/all_data.csv', index_col=0)
    experiment_name = linear_reg_all_data_baseline_negative.__name__
    X = data[['C', 'N']].values
    y = data[['extinction_coefficient']].values / 1000
    scalar = StandardScaler()
    X = scalar.fit_transform(X, y)
    X, y = shuffle(X, y, random_state=random_state)
    X_development, X_validate, y_development, y_validate = train_test_split(X, y, test_size=0.33,
                                                                            random_state=random_state, shuffle=True)
    result = evaluation.evaluate_baseline_negative_model(experiment_name, y_validate)
    return result


def linear_reg_dyomics_data_basic_features():
    data = pd.read_csv('../data/dyomics_data.csv', index_col=0)
    experiment_name = linear_reg_dyomics_data_basic_features.__name__
    X = data[['C', 'H', 'N', 'O']].values
    y = data[['extinction_coefficient']].values / 1000
    scalar = StandardScaler()
    X = scalar.fit_transform(X, y)
    X, y = shuffle(X, y, random_state=random_state)
    X_development, X_validate, y_development, y_validate = train_test_split(X, y, test_size=0.33,
                                                                            random_state=random_state, shuffle=True)
    model = LinearRegression()
    model.fit(X_development, y_development)
    result = evaluation.evaluate_regression_model(experiment_name, X_validate, y_validate, model)
    return result


def linear_reg_dyomics_data_all_features():
    data = pd.read_csv('../data/dyomics_data.csv', index_col=0)
    experiment_name = linear_reg_dyomics_data_all_features.__name__
    X = data[['C', 'H', 'N', 'O', 'S', 'Cl', 'CH3', 'SO3', 'Benzene',
              'C Chain', 'COOH', 'N Rings', 'O Rings', 'C links']].values
    y = data[['extinction_coefficient']].values / 1000
    scalar = StandardScaler()
    X = scalar.fit_transform(X, y)
    X, y = shuffle(X, y, random_state=random_state)
    X_development, X_validate, y_development, y_validate = train_test_split(X, y, test_size=0.33,
                                                                            random_state=random_state, shuffle=True)
    model = LinearRegression()
    model.fit(X_development, y_development)
    result = evaluation.evaluate_regression_model(experiment_name, X_validate, y_validate, model)
    return result

def logit_reg_all_data_all_features():
    data = pd.read_csv('../data/all_data.csv', index_col=0)
    experiment_name = logit_reg_all_data_all_features.__name__
    X = data[['C', 'H', 'N', 'O', 'C Chain']].values
    y = data[['extinction_coefficient']].values / 1000
    scalar = StandardScaler()
    X = scalar.fit_transform(X, y)
    X, y = shuffle(X, y, random_state=random_state)
    y_binary = []
    for i, value in enumerate(y):
        if value < cutoff:
            y_binary.append(0)
        else:
            y_binary.append(1)
    y_binary = np.array(y_binary)
    X_development, X_validate, y_development, y_validate = train_test_split(X, y_binary, test_size=0.33,
                                                                            random_state=random_state)
    # train logistic regression model on dev
    stdX_train = preprocessing.scale(X_development)
    stdX_validate = preprocessing.scale(X_validate)
    model = LogisticRegression(C=0.33, random_state=random_state)
    model.fit(stdX_train, y_development)
    result = evaluation.evaluate_classification_model(experiment_name, stdX_validate, y_validate, model)
    return result

def logit_reg_all_data_basic_features():
    data = pd.read_csv('../data/all_data.csv', index_col=0)
    experiment_name = logit_reg_all_data_basic_features.__name__
    X = data[['C', 'H', 'N', 'O']].values
    y = data[['extinction_coefficient']].values / 1000
    scalar = StandardScaler()
    X = scalar.fit_transform(X, y)
    X, y = shuffle(X, y, random_state=random_state)
    y_binary = []
    for i, value in enumerate(y):
        if value < cutoff:
            y_binary.append(0)
        else:
            y_binary.append(1)
    y_binary = np.array(y_binary)
    X_development, X_validate, y_development, y_validate = train_test_split(X, y_binary, test_size=0.33,
                                                                            random_state=random_state)
    # train logistic regression model on dev
    stdX_train = preprocessing.scale(X_development)
    stdX_validate = preprocessing.scale(X_validate)
    model = LogisticRegression(C=6, random_state=random_state)
    model.fit(stdX_train, y_development)
    result = evaluation.evaluate_classification_model(experiment_name, stdX_validate, y_validate, model)
    return result

def logit_reg_all_data_basic_features_balanced():
    data = pd.read_csv('../data/all_data.csv', index_col=0)
    experiment_name = logit_reg_all_data_basic_features_balanced.__name__
    X = data[['C', 'H', 'N', 'O']].values
    y = data[['extinction_coefficient']].values / 1000
    scalar = StandardScaler()
    X = scalar.fit_transform(X, y)
    X, y = shuffle(X, y, random_state=random_state)
    y_binary = []
    for i, value in enumerate(y):
        if value < cutoff:
            y_binary.append(0)
        else:
            y_binary.append(1)
    y_binary = np.array(y_binary)
    X_development, X_validate, y_development, y_validate = train_test_split(X, y_binary, test_size=0.33,
                                                                            random_state=random_state)
    # train logistic regression model on dev
    stdX_train = preprocessing.scale(X_development)
    stdX_validate = preprocessing.scale(X_validate)
    model = LogisticRegression(C=6, random_state=random_state, class_weight='balanced')
    model.fit(stdX_train, y_development)
    result = evaluation.evaluate_classification_model(experiment_name, stdX_validate, y_validate, model)
    return result