## 문제1: 
#필요한 패키지 가져오기
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = sns.load_dataset('titanic')
df.head()
X_train, X_test, y_train, y_test = train_test_split(df, df['survived'], test_size = 0.2, random_state = 42, stratify = df['survived'])
X_train = X_train.drop(['alive', 'survived'], axis = 1)
X_test = X_test.drop(['alive', 'survived'], axis = 1)

#사용자 코드
# 1. 결측치 확인
missing = ['age']
for m in missing:
    X_train[m] = X_train[m].fillna(X_train[m].mean())
    X_test[m] = X_test[m].fillna(X_test[m].mean())
X_train['deck'] = X_train['deck'].fillna('C')
X_test['deck'] = X_test['deck'].fillna('C')

X_train['embarked'] = X_train['embarked'].fillna('S')
X_test['embarked'] = X_test['embarked'].fillna('S')

X_train['embark_town'] = X_train['embark_town'].fillna('S')
X_test['embark_town'] = X_test['embark_town'].fillna('S')

#print(X_train.isna().sum())
#print()
#print(X_test.isna().sum())

#2. 라벨인코딩
from sklearn.preprocessing import LabelEncoder

label = ['sex', 'embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town', 'alone']
X_train[label] = X_train[label].apply(LabelEncoder().fit_transform)
X_test[label] = X_test[label].apply(LabelEncoder().fit_transform)
#print(X_train.head())

# 3. 카테고리 변수로 변환 / 더미변수로 변환

dtype = ['sex', 'class', 'pclass']
X_train[dtype] = X_train[dtype].astype('category')
X_test[dtype] = X_test[dtype].astype('category')
#print(X_train.dtypes)

#더미변수로 변환
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
#print(X_train.head())

# 4. 파생변수 만들기
X_train['age_qcut'] = pd.qcut(X_train['age'], 5, labels = False)
X_test['age_qcut'] = pd.qcut(X_test['age'], 5, labels = False)

# 5. scaling
#help('sklearn.preprocessing')
from sklearn.preprocessing import MinMaxScaler
min = MinMaxScaler()
scaler = ['age', 'fare']
min.fit(X_train[scaler])
X_train[scaler] = min.transform(X_train[scaler])
X_test[scaler] = min.transform(X_test[scaler])

# 6. 데이터 분리
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42, stratify = y_train)
#print(X_train.shape)
#print(X_valid.shape)

# 7. 모형학습하기, 앙상블
#랜덤포레스트
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()
model1.fit(X_train, y_train)
pred1 = pd.DataFrame(model1.predict_proba(X_valid))
#print(pred1)

from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier()
model2.fit(X_train, y_train)
pred2 = pd.DataFrame(model2.predict_proba(X_valid))
#print(pred2)

from sklearn.ensemble import VotingClassifier
model3 = VotingClassifier(estimators = [('logistic', model1), ('random', model2)], voting = 'soft')
model3.fit(X_train, y_train)
pred3 = pd.DataFrame(model3.predict_proba(X_valid))

#8. 모형평가 roc-auc
from sklearn.metrics import roc_auc_score
#print('로지스틱', roc_auc_score(y_valid, pred1.iloc[:, 1]))
#print('랜덤포레스트', roc_auc_score(y_valid, pred2.iloc[:, 1]))
#print('voting', roc_auc_score(y_valid, pred3.iloc[:, 1]))
#help('sklearn.metrics')

#9. 파일저장
result = pd.DataFrame(model3.predict_proba(X_test))
result = result.iloc[:, 1]
pd.DataFrame({'id': X_test.index, 'result': result}).to_csv('00300.csv', index = False)
#print(result)
#result = result.iloc[:, 1]
example = pd.read_csv('00300.csv')
print(example.head())



