import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import r2_score, roc_curve, roc_auc_score, precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve

cutoff = 150
random_state = 12


def evaluate_regression_model(experiment_name, X_validate, y_validate, model):
    """
    use for validating regression models.
    :param experiment_name: name of the experiment
    :param X_validate: features validate set
    :param y_validate: target validate set
    :param model: trained model
    :return: dataframe consisting of results of validation testing
    """
    y_pred = model.predict(X_validate)
    r2 = r2_score(y_validate, y_pred)
    y_class_val = []
    for i, value in enumerate(y_validate):
        if value < cutoff:
            y_class_val.append(0)
        else:
            y_class_val.append(1)
    y_class_pred = []
    for i, value in enumerate(y_pred):
        if value < cutoff:
            y_class_pred.append(0)
        else:
            y_class_pred.append(1)
    std_pred_y = preprocessing.scale(y_pred)
    fpr, tpr, thresholds = roc_curve(y_class_val, std_pred_y, drop_intermediate=False)
    accuracy = accuracy_score(y_class_val, y_class_pred)
    precision = precision_score(y_class_val, y_class_pred)
    recall = recall_score(y_class_val, y_class_pred)
    f1 = f1_score(y_class_val, y_class_pred)
    auc = roc_auc_score(y_class_val, std_pred_y)
    coefficients = model.coef_
    result = pd.DataFrame({'experiment': experiment_name, 'r2': r2, 'accuracy': accuracy, 'precision': precision,
                           'recall': recall, 'f1': f1, 'auc': auc, 'fpr': [fpr], 'tpr': [tpr],
                           'thresholds': [thresholds], 'y_pred': [y_pred.flatten()], 'y_val': [y_validate.flatten()],
                           'coefficients': [coefficients.flatten()]})
    result.set_index('experiment', inplace=True)
    return result


def evaluate_classification_model(experiment_name, stdX_validate, y_validate, model):
    """
    use for validating classification models.
    :param experiment_name: name of the experiment
    :param stdX_validate: features validate set
    :param y_validate: target validate set
    :param model: trained model
    :return: dataframe consisting of results of validation testing
    """
    # predict val
    y_pred = model.predict(stdX_validate)
    y_probs = model.predict_proba(stdX_validate)[:, 1]
    # get classification testing on val
    accuracy = accuracy_score(y_validate, y_pred)
    precision = precision_score(y_validate, y_pred)
    recall = recall_score(y_validate, y_pred)
    f1 = f1_score(y_validate, y_pred)
    auc = roc_auc_score(y_validate, y_probs)
    fpr, tpr, thresholds = roc_curve(y_validate, y_probs, drop_intermediate=False)
    coefficients = model.coef_


    precision_curve, recall_curve, _ = precision_recall_curve(y_validate, y_probs)
    # retrieve probability of being 1(in second column of probs_y)

    result = pd.DataFrame({'experiment': experiment_name, 'accuracy': accuracy, 'precision': precision,
                           'recall': recall, 'f1': f1, 'auc': auc, 'fpr': [fpr], 'tpr': [tpr],
                           'thresholds': [thresholds], 'coefficients': [coefficients.flatten()]})
    result.set_index('experiment', inplace=True)
    return result


def evaluate_baseline_negative_model(experiment_name, y_validate):
    """
    use for validating regression models.
    :param experiment_name: name of the experiment
    :param X_validate: features validate set
    :param y_validate: target validate set
    :param model: trained model
    :return: dataframe consisting of results of validation testing
    """
    y_class_val = []
    y_class_pred = []
    for i, value in enumerate(y_validate):
        y_class_pred.append(0)
        if value < cutoff:
            y_class_val.append(0)
        else:
            y_class_val.append(1)
    fpr, tpr, thresholds = roc_curve(y_class_val, y_class_pred, drop_intermediate=False)
    accuracy = accuracy_score(y_class_val, y_class_pred)
    precision = precision_score(y_class_val, y_class_pred)
    recall = recall_score(y_class_val, y_class_pred)
    f1 = f1_score(y_class_val, y_class_pred)
    auc = roc_auc_score(y_class_val, y_class_pred)
    result = pd.DataFrame({'experiment': experiment_name, 'accuracy': accuracy, 'precision': precision,
                           'recall': recall, 'f1': f1, 'auc': auc, 'fpr': [fpr], 'tpr': [tpr],
                           'thresholds': [thresholds]})
    result.set_index('experiment', inplace=True)
    return result
