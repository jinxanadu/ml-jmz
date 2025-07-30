# 随堂练习： 模拟把相关代码弄到线上环境（非开发环境），并在线上环境可以加载模型并预测数据
# 新建一个.py文件，假装这个文件
import pandas as pd
import numpy as np
from pathlib import Path

housing = pd.read_csv(Path("datasets/housing/housing.csv"))
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
#%%
from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])
#%%
from sklearn.model_selection import train_test_split

strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)
#%%
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
housing = strat_train_set.copy()
#%%
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
import joblib

# 假设这个单元格的代码不是在这台电脑上，而是在运行模型的服务器
# 下面代码省略完整的


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted, check_array


def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

class NearestLabel(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):  # 即使不用y，也需要它
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()

        X = check_array(X)  # 检查X是不是数组
        self.n_features_in_ = X.shape[1]  # 所有估计器会把输入特征的数量存下来
        self.knn_ = KNeighborsRegressor(n_neighbors=self.n_neighbors)
        self.knn_.fit(X, y)
        return self  # 永远返回 self!

    def transform(self, X):
        check_is_fitted(self)  # 检查是否适配过数据 （检查是否有那些下划线结尾的参数)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]

        return self.knn_.predict(X).reshape(-1, 1)

    def get_feature_names_out(self, names=None):
        return ["Nearest Neighbors Median Housing Price"]

final_model_reloaded = joblib.load("./models/my_california_housing_model.pkl")

new_data = housing.iloc[:5]  # pretend these are new districts
predictions = final_model_reloaded.predict(new_data)
print(predictions)