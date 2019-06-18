import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)



'''Delete all the anomaly data for the GarageArea field (for the same data set in the use case: House Prices)
.* for this task you need to plot GaurageArea field and SalePrice in scatter plot, then check which numbers are
anomalies'''

'''Read the data from the csv file '''
data_set = pd.read_csv('./lessonplan5.csv')

'''Set the x and y (independent and dependent values for the data set)'''
data_set_y = data_set.SalePrice.values.reshape(-1, 1)
data_set_x = data_set.GarageArea.values.reshape(-1, 1)

'''Linear Regression Model'''
data_linear = LinearRegression().fit(data_set_x, data_set_y)

'''Prediction Value'''
y_predictor = data_linear.predict(data_set_x)


'''Display and plot linear regression data'''
plt.xlabel('Garage Area')
plt.ylabel('Sale Price')
plt.title('Linear Regression w/ Anomalies')
plt.scatter(data_set_x, data_set_y)
plt.plot(data_set_x, y_predictor, color='orange')
plt.show()

'''Delete Anomalies '''
data_linear_nonoutliers = data_set[(np.abs(stats.zscore(data_set.GarageArea)) < 3)]
data_linear_nonoutliers = data_linear_nonoutliers[(data_linear_nonoutliers.GarageArea != 0)]

'''Set the x and y (independent and dependent values for the data set)'''
data_linear_x = data_linear_nonoutliers.GarageArea
data_linear_y = data_linear_nonoutliers.SalePrice


plt.xlabel('Garage Area')
plt.ylabel('Sale Price')
plt.title('Linear Regression w/o Anomalies')
plt.scatter(data_linear_x, data_linear_y)
plt.plot(data_set_x, y_predictor, color='yellow')
plt.show()



'''Create Multiple Regression for the “wine quality” dataset. In this data set “quality” is the target label.
Evaluate the model using RMSE and R2 score. **You need to delete the null values in the data set
**You need to find the top 3 most correlated features to the target label(quality)'''

'''Read the data from the csv file'''
wine_quality = pd.read_csv('winequality-red.csv')

'''Number of features in the wine quality file'''
features = wine_quality.select_dtypes(include=[np.number])

'''Correlation'''
correlation = features.corr()

'''Output to screen correlation'''
print(correlation)

wine_quality_x = wine_quality.drop('quality', axis=1)
wine_quality_y = wine_quality.quality

x_train, x_test, y_train, y_test = train_test_split(wine_quality_x, wine_quality_y, random_state=42, test_size=.29)

'''Linear Regression & Model'''
wine_linear = LinearRegression()
wine_model = wine_linear.fit(x_train, y_train)

'''Performance Evaluation'''
print("R^2 is: \n", wine_model.score(x_test, y_test))
predictions = wine_model.predict(x_test)
print('RMSE is: \n', mean_squared_error(y_test, predictions))
print()


'''Linear Regression & Model Based on 3 highest correlated features'''
new_wine_quality_x = wine_quality[['sulphates', 'alcohol', 'volatile acidity']]
new_wine_quality_y = wine_quality.quality


x_train, x_test, y_train, y_test = train_test_split(wine_quality_x, wine_quality_y, random_state=38, test_size=.32)

'''Linear Regression & Model Based on 3 highest correlated features'''
new_wine_linear = LinearRegression()
new_wine_model = new_wine_linear.fit(x_train, y_train)

'''Performance Evaluation Based on 3 highest correlated features'''
print("R^2 is: \n", new_wine_model.score(x_test, y_test))
predictions = new_wine_model.predict(x_test)

print('RMSE is: \n', mean_squared_error(y_test, predictions))








