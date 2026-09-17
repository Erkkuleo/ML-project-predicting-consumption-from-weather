import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.tree import export_text, plot_tree, DecisionTreeRegressor, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

features = ['temp_c', "wind_ms", "humid_pc", "hour_sin", "hour_cos", "weekday", "month"]

def read_data():
    df = pd.read_csv('data/train.csv')

    df_test = pd.read_csv('data/test.csv')

    df.drop(columns=['time_utc', 'local_time', 'is_weekend', 'hour'], inplace=True)
    df_test.drop(columns=['time_utc', 'local_time', 'is_weekend', 'hour'], inplace=True)

    X = df[features].to_numpy()
    y = df['consumption_mwh'].to_numpy()

    X_test = df_test[features].to_numpy()
    y_test = df_test['consumption_mwh'].to_numpy()

    return X, y, X_test, y_test