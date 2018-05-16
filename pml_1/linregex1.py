import pandas as pd
import matplotlib.pyplot as plt

# give the path to the file 
path_to_file = "./data/ex1data1.csv"
data = pd.read_csv(path_to_file, header=None, names=["Population", "Profit"])

x = data["Population"].values.reshape(-1, 1)
y = data["Profit"]
## do a scatter plot with data x and y
plt.figure(figsize=(9, 6))
plt.scatter(x, y, c='k')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.show()

#### Your code here
from sklearn import linear_model
ols = linear_model.LinearRegression()

ols.fit(x, y)

plt.figure(figsize=(9, 6))
plt.plot(x, ols.predict(x), c='r', lw=1.5, label='Predicted relation')
plt.scatter(x, y, c='k')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.show()

print("beta_1: %.3f" %ols.coef_)
print("beta_0: %.3f" %ols.intercept_)
print("RSS: %.2f" % sum((y - ols.predict(x)) ** 2))
## score: the R^2 of the fitted model
print('R^2: %.2f' % ols.score(x, y))