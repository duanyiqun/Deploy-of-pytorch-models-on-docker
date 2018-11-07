from sklearn import metrics
import pandas as pd 

df = pd.read_csv('/Users/duanyiqun/Downloads/20K_testresult_cat.csv')
print(df['porn'])
x_predicted = df['porn']
x_label = df['TRUE']
print(x_label)
#auc_score = metrics.auc(x_label,x_predicted)
#print(auc_score)
print('for catnet')
fpr, tpr, thresholds = metrics.roc_curve(x_label,x_predicted, pos_label=1)
print(fpr)
print(tpr)
auc_score = metrics.auc(fpr,tpr)
print(auc_score)

df2 = pd.read_csv('/Users/duanyiqun/Downloads/20K_testresult.csv')
x_predicted = df2['porn']
print('for pretrained dense')
fpr, tpr, thresholds = metrics.roc_curve(x_label,x_predicted, pos_label=1)
print(fpr)
print(tpr)
auc_score = metrics.auc(fpr,tpr)
print(auc_score)

df2 = pd.read_csv('/Users/duanyiqun/Downloads/20K_testresult_sdmn.csv')
x_predicted = df2['porn']
print('for sdmn')
fpr, tpr, thresholds = metrics.roc_curve(x_label,x_predicted, pos_label=1)
print(fpr)
print(tpr)
auc_score = metrics.auc(fpr,tpr)
print(auc_score)
