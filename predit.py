import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn.linear_model import LogisticRegression


def load_data(path):
    data_train = pd.read_csv(path, index_col='PassengerId')
    return data_train


def check_data(data_train):
    data_train.info()  # 查看数据完整性
    id_list = data_train.index.tolist()

    age_list = []
    data_age = data_train.Age
    for id in id_list:
        age = data_age.loc[id]
        if not pd.isnull(age):
            age = age // 10
        age_list.append(age)
    data_train.Age = age_list

    fare_list = []
    data_fare = data_train.Fare
    for id in id_list:
        fare = data_fare.loc[id]
        if not pd.isnull(fare):
            fare = fare // 20
        fare_list.append(fare)
    data_train.Fare = fare_list

    data_train.loc[(data_train.Cabin.notnull()), 'Cabin'] = "Yes"
    data_train.loc[(data_train.Cabin.isnull()), 'Cabin'] = "No"

    # 查看乘客等级pclass对是否幸存的影响
    figshow('Pclass', data_train, 'passenger class', 'number of people')

    # 查看性别sex对是否幸存的影响
    figshow('Sex', data_train, 'sex', 'number of people')

    # 查看年龄age对是否幸存的影响
    data_train_age = data_train
    for id in id_list:
        age = data_train_age['Age'][id]
        if pd.isnull(age):
            data_train_age.drop(id, axis=0, inplace=True)
    figshow('Age', data_train_age, 'age', 'number of people')

    # 查看兄弟姐妹/配偶sibsp对是否幸存的影响
    figshow('SibSp', data_train, 'siblings / spouses', 'number of people')

    # 查看父母/孩子parch对是否幸存的影响
    figshow('Parch', data_train, 'parents / children', 'number of people')

    # 查看票价fare对是否幸存的影响
    figshow('Fare', data_train, 'fare', 'number of people')

    # 查看船仓号cabin对是否幸存的影响
    figshow('Cabin', data_train, 'cabin', 'number of people')

    # 查看登场港口embarked对是否幸存的影响
    figshow('Embarked', data_train, 'embarked', 'number of people')

    plt.show()


def figshow(c, df, s1, s2):
    locals()['survived_' + c + '_0'] = df[c][df.Survived == 0].value_counts()
    locals()['survived_' + c + '_1'] = df[c][df.Survived == 1].value_counts()
    locals()['df_' + c] = pd.DataFrame({'not survived': locals()['survived_' + c + '_0'],
                                        'survived': locals()['survived_' + c + '_1']})
    locals()['df_' + c].plot(kind='bar', stacked=True)
    plt.xlabel(s1)
    plt.ylabel(s2)
    x = np.arange(len(locals()['df_' + c].index))
    y0 = np.array(locals()['df_' + c]['not survived'].tolist())
    for i in range(len(y0)):
        if pd.isnull(y0[i]):
            y0[i] = 0
    y1 = np.array(locals()['df_' + c]['survived'].tolist())
    for i in range(len(y1)):
        if pd.isnull(y1[i]):
            y1[i] = 0
    y = y0 + y1
    i = 0
    for a, b in zip(x, y):
        p = float(y1[i]) / y[i]
        plt.text(a, b + 0.05, '{:.2%}'.format(p), ha='center', va='bottom', fontsize=7)
        i += 1


def preprocess_data(data):
    data.loc[(data['Cabin'].notnull()), 'Cabin'] = "Yes"
    data.loc[(data['Cabin'].isnull()), 'Cabin'] = "No"

    data.loc[(data['Fare'].isnull()), 'Fare'] = 0

    data_age = data[['Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = data_age[data_age['Age'].notnull()].as_matrix()
    unknown_age = data_age[data_age['Age'].isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1:])

    # 用得到的预测结果填补原缺失数据
    data.loc[(data['Age'].isnull()), 'Age'] = predictedAges

    dummies_Pclass = pd.get_dummies(data['Pclass'], prefix='Pclass')
    dummies_Sex = pd.get_dummies(data['Sex'], prefix='Sex')
    dummies_Cabin = pd.get_dummies(data['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(data['Embarked'], prefix='Embarked')

    data = pd.concat([data, dummies_Pclass, dummies_Sex,
                      dummies_Cabin, dummies_Embarked, ], axis=1)
    data.drop(['Pclass', 'Name', 'Sex', 'Ticket',
               'Cabin', 'Embarked'], axis=1, inplace=True)

    scaler = preprocessing.StandardScaler()
    data_age = data['Age'].values.reshape(-1, 1)
    age_scale = scaler.fit(data_age)
    data['Age_scaled'] = age_scale.transform(data_age)
    data_fare = data['Fare'].values.reshape(-1, 1)
    fare_scale = scaler.fit(data_fare)
    data['Fare_scaled'] = fare_scale.transform(data_fare)

    return data


def LR(data):
    # 用正则取出我们要的属性值
    data = data.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    np_train = data.as_matrix()

    # y即Survival结果
    y = np_train[:, 0]

    # X即特征属性值
    X = np_train[:, 1:]

    # fit到RandomForestRegressor之中
    lr = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    lr.fit(X, y)
    return lr


def predit(data, lr):
    data = data.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    predictions = lr.predict(data)
    result = pd.DataFrame({'PassengerId': data.index, 'Survived': predictions.astype(np.int32)})
    return result


def main():
    path_train = 'data/train.csv'
    data_train = load_data(path_train)
    flag = 0
    if not flag:
        check_data(data_train)
    else:
        data_train = preprocess_data(data_train)
        lr = LR(data_train)
        path_test = 'data/test.csv'
        data_test = load_data(path_test)
        data_test = preprocess_data(data_test)
        result = predit(data_test, lr)
        save_path = 'data/predit.csv'
        result.to_csv(save_path, index=False)


if __name__ == '__main__':
    main()
