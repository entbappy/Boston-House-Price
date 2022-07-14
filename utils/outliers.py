import pandas as pd
from sklearn.datasets import load_boston

data = load_boston()
X = pd.DataFrame(data.data, columns= data.feature_names)

class RemoveOutliers:
    def __init__(self):
        pass

    def remove(self, data):
        test_data = data
        for col in data.columns:
            ignore = ['CHAS']
            if col not in ignore:
                IQR = X[col].quantile(0.75) - X[col].quantile(0.25)
                lower_bridge = X[col].quantile(0.25) - (IQR * 1.5)
                upper_bridge = X[col].quantile(0.75) + (IQR * 1.5)

                test_data.loc[test_data[col] >= upper_bridge, col] = upper_bridge
                test_data.loc[test_data[col] <= lower_bridge, col] = lower_bridge

        return test_data

