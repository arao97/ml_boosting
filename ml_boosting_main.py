import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from df_cleaning import preprocessor as pp


def accuracy(model, input_test, output_test):
    y_pred = model.predict(input_test)
    predictions = [round(value) for value in y_pred]
    return accuracy_score(output_test, predictions)

# Reading dataset
df = pd.read_csv('./bank_data/bank-additional-full.csv', na_values=['NA'])
columns = df.columns.values[0].split(';')
data = df.values

customer_details = list()
for customer in data:
    customer_details.append(customer[0].split(';'))
    if len(customer[0].split(';')) != 21:
        print(len(customer[0].split(';')))
customer_df = pd.DataFrame(customer_details, columns=columns)

# Preprocessing by one-hot encoding
encoded_customer_df = pp(customer_df)

X = encoded_customer_df.drop(['"y"'], axis=1).values
Y = encoded_customer_df['"y"'].values

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)

# Adaboost model training
start = time.time()
ada_model = AdaBoostClassifier(n_estimators=120)
ada_model.fit(x_train, y_train)
end = time.time()
ada_y_prob = ada_model.predict_proba(x_test)[:, 1]

print("Accuracy of the AdaBoost model = %.3f%%" % (accuracy(ada_model, x_test, y_test) * 100))
print("Time taken to fit model = %.3f seconds" % (end - start))
ada_fpr, ada_tpr, _ = roc_curve(y_test, ada_y_prob)
ada_roc_auc = auc(ada_fpr, ada_tpr)

# xGBoost model training
start = time.time()
xg_model = XGBClassifier()
xg_model.fit(x_train, y_train)
end = time.time()
xg_y_pred = xg_model.predict_proba(x_test)[:, 1]

print("Accuracy of the xGBoost model = %.3f%%" % (accuracy(xg_model, x_test, y_test) * 100))
print("Time taken to fit model = %.3f seconds" % (end - start))
xg_fpr, xg_tpr, _ = roc_curve(y_test, xg_y_pred)
xg_roc_auc = auc(xg_fpr, xg_tpr)

#Plotting
plt.figure(1)
plt.plot(ada_fpr, ada_tpr, color='red', lw=2, label='AdaBoost(AUC = %0.3f)' % ada_roc_auc)
plt.plot(xg_fpr, xg_tpr, color='blue', lw=2, label='xGBoost(AUC = %0.3f)' % xg_roc_auc)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristics')
plt.legend(loc='lower right')
plt.show()
