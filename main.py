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
import src.data as da

X, y, X_test, y_test = da.read_data()
which_to_run = None

while (which_to_run != "3"):
    print("Do you want to run polynomial regression (1), decision tree (2) exit (3): \n")

    which_to_run = input()

    if (which_to_run == "1"):

        poly = PolynomialFeatures(degree= 3)
        X_poly = poly.fit_transform(X)

        lin_regr = LinearRegression(fit_intercept=False)
        lin_regr.fit(X_poly, y)

        y_pred = lin_regr.predict(X_poly)
        r2 = r2_score(y,y_pred)
        print("R2 of polynomial regression : ", r2)


        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlabel("prediction from all 7 features (MWh)")
        ax.set_ylabel("actual consumption (MWh)")
        ax.set_title(f"Polynomial regression (degree 3) on all features, R2 = {r2:.2f}")
        ax.scatter(y_pred, y, s=2, alpha=0.2, c="skyblue", label="training datapoints")
        ax.plot([y.min(), y.max()], [y.min(), y.max()], color='r', linewidth=2, label="the fit (prediction = actual)")
        ax.legend()
        plt.show()

    elif (which_to_run == "2"):
    #====================================#
    # Decision tree part -> this should maybe be changed to random forest in future
    #====================================#

        clf_tree = DecisionTreeRegressor(random_state=0, max_depth=10)
        clf_tree.fit(X, y)

        y_tree_pred = clf_tree.predict(X_test)
        print("R2 of decision tree : ", r2_score(y_test, y_tree_pred))


        plt.figure(figsize=(16, 8))
        plot_tree(clf_tree, feature_names=da.features, filled=True, max_depth=3, fontsize=8)
        plt.show()
