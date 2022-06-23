## 문제2:회귀문제. 종속변수 mpg
# 제출한 모형의 성능은 RMSE, MAE평가지표에 따라 채점

#필요한 패키지 가져오기
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df = sns.load_dataset('mpg')
X_train, X_test, y_train, y_test = train_test_split(df, df['mpg'], test_size = 0.2, random_state = 42)
X_train = X_train.drop(['mpg'], axis = 1)
X_test = X_test.drop(['mpg'], axis = 1)

#사용자 코드
# 1. 결측치 확인
#print(X_train.head())
#범주형변수: origin
#연속형변수: cylinders, displacement, horsepower, weight, acceration, model_year
#qcut으로 model_year 나누면 좋을 듯
X_train['horsepower'] = X_train['horsepower'].fillna(X_train['horsepower'].median())
X_test['horsepower'] = X_test['horsepower'].fillna(X_test['horsepower'].median())

#print(X_train.isna().sum())

#2. 라벨인코딩
#print(X_train['name'].value_counts())
from sklearn.preprocessing import LabelEncoder
labels = ['origin']
#help('sklearn.preprocessing.LabelEncoder')
X_train[labels] = X_train[labels].apply(LabelEncoder().fit_transform)
X_test[labels] = X_test[labels].apply(LabelEncoder().fit_transform)
#print(X_train.head())

#3. 카테고리변수 변환, 더미변수로 변환
category = ['origin']
X_train[category] = X_train[category].astype('category')
X_test[category] = X_test[category].astype('category')

#더미변수로 변환
#print(X_train.dtypes)
#print(X_train['cylinders'].value_counts())
X_train = X_train.drop(['name'], axis = 1)
X_test = X_test.drop(['name'], axis = 1)
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

#4. 새로운 변수 만들기
X_train['model_year_qcut'] = pd.qcut(X_train['model_year'], 5, labels = False)
X_test['model_year_qcut'] = pd.qcut(X_test['model_year'], 5, labels = False)

#5. scaling
from sklearn.preprocessing import MinMaxScaler
scaler = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']
# displacement, horsepower, weight
min = MinMaxScaler()
min.fit(X_train[scaler])
X_train[scaler] = min.transform(X_train[scaler])
X_test[scaler] = min.transform(X_test[scaler])
#print(X_train.head())

# 6. 데이터 나누기
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)
#print(X_train.shape)
#print(X_valid.shape)

#7. 모델 적용하기 - 회귀문제
#linear regression
from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model1.fit(X_train, y_train)
pred1 = model1.predict(X_valid)
#print(pred1)

# randomforestregressor
from sklearn.ensemble import RandomForestRegressor
model2 = RandomForestRegressor()
model2.fit(X_train, y_train)
pred2 = model2.predict(X_valid)

# 앙상블(스태킹)
from sklearn.ensemble import StackingRegressor
estimators = [('lr', model1), ('rf', model2)]
model3 = StackingRegressor(estimators = estimators, final_estimator = RandomForestRegressor())
model3.fit(X_train, y_train)
pred3 = model3.predict(X_valid)

#8. 모델 평가하기
#MES
from sklearn.metrics import mean_squared_error
#print('선형회귀MSE', mean_squared_error(y_valid, pred1))
#print('랜포MSE', mean_squared_error(y_valid, pred2))
#print('스태킹MSE', mean_squared_error(y_valid, pred3))

#RMSE
#print('선형회귀MSE', np.sqrt(mean_squared_error(y_valid, pred1)))
#print('랜포MSE', np.sqrt(mean_squared_error(y_valid, pred2)))
#print('스태킹MSE', np.sqrt(mean_squared_error(y_valid, pred3)))

#9. 문서로 저장하기
#print(X_test.shape)
#print(X_test.head())
result = pd.DataFrame(model3.predict(X_test))
#print(result)
result = result.iloc[:, 0]
pd.DataFrame({'id':X_test.index, 'result': result}).to_csv('0003000.csv', index = False)

# 확인
check = pd.read_csv('0003000.csv')
print(check.head())