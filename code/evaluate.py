from sklearn import metrics

def overall_auc(y_true, y_pred):
    return metrics.roc_auc_score(y_true,y_pred)

