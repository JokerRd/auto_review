from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd


def load_data():
    iris = fetch_ucirepo(id=53)

    iris_data = iris.data.features
    iris_target = iris.data.targets

    return train_test_split(iris_data, iris_target)


def preprocess(X_train, X_test):
    X_scaled_train = pd.DataFrame(StandardScaler().fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_scaled_test = pd.DataFrame(StandardScaler().fit_transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_scaled_train, X_scaled_test


def create_model(X_train, y_train):
    reg_model = LogisticRegression()
    reg_model.fit(X_train, y_train)
    return reg_model


def test_model(X_test, y_test, reg_model):
    y_pred = reg_model.predict(X_test)
    return accuracy_score(y_test, y_pred)


X_train, X_test, y_train, y_test = load_data()
X_train, X_test = preprocess(X_train, X_test)
reg_model = create_model(X_train, y_train)
print("Accuracy Score: ", test_model(X_test, y_test, reg_model))