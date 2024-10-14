import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score

data = pd.read_csv(r"/Users/siddu/Desktop/MY Projects/Basics-of-ML-master/insurance.csv")
print(data.head(15))

sex = data.iloc[:,1:2].values
smoker = data.iloc[:,4:5].values

le = LabelEncoder()
sex[:,0] = le.fit_transform(sex[:,0])
sex = pd.DataFrame(sex)
sex.columns = ['sex']
le_sex_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Sklearn label encoding for sex")
print(le_sex_mapping)
print(sex[:10])

smoker[:,0] = le.fit_transform(smoker[:,0])
smoker = pd.DataFrame(smoker)
smoker.columns = ['smoker']
le_smoker_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

region = data.iloc[:,5:6].values
ohe = OneHotEncoder()
region = ohe.fit_transform(region).toarray()
region = pd.DataFrame(region)
region.columns = ['northeast', 'northwest', 'southeast', 'sothwest']
print("Sklearn one hot encoding for region")
print(region[:10])

X_num = data[['age','bmi','children']]
X_final = pd.concat([X_num, sex, smoker, region], axis=1)

y_final = data[['charges']].copy()

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size= 0.33, random_state=0)

s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(np.float64))
X_test = s_scaler.transform(X_test.astype(np.float64))

print(pd.DataFrame(X_test))         

lr = LinearRegression().fit(X_train, y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

print(f"lr.coef_:{lr.coef_}")
print(f"lr.intercept_: {lr.intercept_}")
print('lr train score %.3f, lr test score %.3f' % (lr.score(X_train,y_train), lr.score(X_test,y_test)))

poly = PolynomialFeatures(degree = 2)

X_poly = poly.fit_transform(X_final)


X_train, X_test, y_train, y_test = train_test_split(X_poly, y_final, test_size= 0.33, random_state=0)

s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(np.float64))
X_test = s_scaler.transform(X_test.astype(np.float64))

poly_lr = LinearRegression().fit(X_train, y_train)
y_train_pred = poly_lr.predict(X_train)
y_test_pred = poly_lr.predict(X_test)

print('poly train score %.3f, poly test score %.3f' % (poly_lr.score(X_train,y_train), poly_lr.score(X_test,y_test)))

svr = SVR(kernel = 'linear', C=300)

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size= 0.33, random_state=0)

s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(np.float64))
X_test = s_scaler.transform(X_test.astype(np.float64))

svr = svr.fit(X_train, y_train.values.ravel())
y_train_pred = svr.predict(X_train)
y_test_pred = svr.predict(X_test)

print('svr train score %.3f, svr test score %.3f' % (svr.score(X_train,y_train), svr.score(X_test,y_test)))

dt = DecisionTreeRegressor(random_state = 0)

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size= 0.33, random_state=0)

s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(np.float64))
X_test = s_scaler.transform(X_test.astype(np.float64))

dt = dt.fit(X_train, y_train.values.ravel())
y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)

print('dt train score %.3f, dt test score %.3f' % (dt.score(X_train,y_train), dt.score(X_test,y_test)))

forest= RandomForestRegressor(n_estimators = 100, criterion = 'squared_error', random_state = 1, n_jobs=-1)

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size= 0.33, random_state=0)

s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(np.float64))
X_test = s_scaler.transform(X_test.astype(np.float64))

forest.fit(X_train, y_train.values.ravel())
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

print('forest train score %.3f, forest test score %.3f' % (forest.score(X_train,y_train), forest.score(X_test,y_test)))
