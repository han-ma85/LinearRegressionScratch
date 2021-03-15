from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from linear_regression import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

X,y=datasets.make_regression(n_samples=100,n_features=2,noise=20,random_state=4)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)
'''print(X_train.shape)
print(y_train.shape)

##Visualization of data points
plt.interactive(False)
fig=plt.figure(figsize=(10,6))
plt.scatter(X[:,0],y,color='r',marker='o',s=30)
plt.show()
'''

regressor=LinearRegression(lr=0.01)
regressor.fit(X_train,y_train)
prediction=regressor.predict(X_test)

def mean_squared_error(y_true,y_predict):
    mse=np.mean((y_true-y_predict)**2)
    return mse

mse=mean_squared_error(y_test,prediction)
print(mse)

score= r2_score(y_test,prediction)
print(score)

'''
## ploting-work for simple LR(n_features=1)
y_pred_line=regressor.predict(X)
fig=plt.figure(figsize=(10,6))
plt.scatter(X[:,0],y,color='r',marker='o',s=30)
plt.plot(X,y_pred_line,color='b')
plt.show()'''