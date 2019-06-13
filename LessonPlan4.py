import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split



'''Read the data from the csv file '''
data_set = pd.read_csv('./glass.csv')
data_set_y = data_set.Type


data_set_columns = data_set.columns.values
data_set_features = data_set.drop(['Type'], axis=1)


'''Implementation of the Naive Bayes Method'''
data_set_features, x_test, data_set_y, y_test = train_test_split(data_set_features, data_set_y, test_size=0.2)

model = GaussianNB()
model.fit(data_set_features, data_set_y)


print("Naive Bayes:", model.score(x_test, y_test))

'''Implementation of the SVM Method'''
svc = SVC(kernel="linear", gamma="auto")
svc.fit(data_set_features, data_set_y)
Y_pred = svc.predict(data_set_features)
acc_svc = round(svc.score(data_set_features, data_set_y) * 100, 2)
print("SVM accuracy for LINEAR:", acc_svc)
svc_2 = SVC(kernel="rbf", gamma="auto")
svc_2.fit(data_set_features, data_set_y)
Y_pred = svc_2.predict(data_set_features)
acc_svc = round(svc_2.score(data_set_features, data_set_y) * 100, 2)
print("SVM accuracy for RBF:", acc_svc)



