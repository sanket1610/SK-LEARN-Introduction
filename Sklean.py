
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

def boston_reg():
    boston = load_boston()
    x = boston['data']
    y = boston['target']
    lr = LinearRegression()
    print(boston.keys())
    x = boston['data']
    y = boston['target']
    x_train, x_test, y_train, y_test =  train_test_split(x,y, random_state= 1, test_size = 0.3)
    lr.fit(x_train,y_train)
    prediction = lr.predict(x_test)

    plt.scatter(prediction, y_test)
    plt.show()
    df = pd.DataFrame(boston.feature_names,lr.coef_)
    return df


if __name__ == "__main__":
    boston_reg()

# Looking at the dataframe we can say NOX has most influence on price of the housing in Boston