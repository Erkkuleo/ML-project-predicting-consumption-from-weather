import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.tree import export_text, plot_tree, DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import src.data as da

X, y, X_test, y_test, X_val, y_val = da.read_data()
which_to_run = None

while (which_to_run != "0"):
    print("Do you want to run: \n polynomial regression (1) \n decision tree (2) \n random forest (3) \n exit (0): \n")

    which_to_run = input()

    if (which_to_run == "1"):
        # degree 1 is plain linear regression
        best_r2_val = -np.inf
        best_degree = None

        for degree in range(1, 5):
            poly = PolynomialFeatures(degree=degree)
            lin_regr = LinearRegression(fit_intercept=False)
            lin_regr.fit(poly.fit_transform(X), y)

            # use transform (not fit_transform) on val/test so they get the same feature mapping as train
            r2_val = r2_score(y_val, lin_regr.predict(poly.transform(X_val)))
            print(f"degree = {degree}: validation R2 = {r2_val:.4f}")

            if r2_val > best_r2_val:
                best_r2_val = r2_val
                best_degree = degree
                best_poly = poly
                best_lin_regr = lin_regr

        print(f"\nBest: degree = {best_degree}")
        print("R2 of polynomial regression on training: ", r2_score(y, best_lin_regr.predict(best_poly.transform(X))))
        print("R2 of polynomial regression on validation: ", best_r2_val)
        y_pred = best_lin_regr.predict(best_poly.transform(X_test))
        r2 = r2_score(y_test, y_pred)
        print("R2 of polynomial regression on test: ", r2)
        print("MAPE of polynomial regression on test: ", mean_absolute_percentage_error(y_test, y_pred), "\n")


        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlabel("prediction from all 7 features (MWh)")
        ax.set_ylabel("actual consumption (MWh)")
        ax.set_title(f"Polynomial regression (degree {best_degree}) on all features, test R2 = {r2:.2f}")
        ax.scatter(y_pred, y_test, s=2, alpha=0.2, c="skyblue", label="test datapoints")
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='r', linewidth=2, label="the fit (prediction = actual)")
        ax.legend()
        plt.show()

    elif (which_to_run == "2"):

        best_r2_val = -np.inf
        best_depth = None

        for depth in range(2, 21):
            tree = DecisionTreeRegressor(random_state=0, max_depth=depth)
            tree.fit(X, y)

            r2_val = r2_score(y_val, tree.predict(X_val))
            print(f"max_depth = {depth}: validation R2 = {r2_val:.4f}")

            if r2_val > best_r2_val:
                best_r2_val = r2_val
                best_depth = depth
                clf_tree = tree

        print(f"\nBest: max_depth = {best_depth}")
        print("R2 of decision tree on training: ", r2_score(y, clf_tree.predict(X)))
        print("R2 of decision tree on validation: ", best_r2_val)
        y_tree_pred = clf_tree.predict(X_test)
        print("R2 of decision tree on test: ", r2_score(y_test, y_tree_pred))
        print("MAPE of decision tree on test: ", mean_absolute_percentage_error(y_test, y_tree_pred), "\n")


        plt.figure(figsize=(16, 8))
        plot_tree(clf_tree, feature_names=da.features, filled=True, max_depth=3, fontsize=8)
        plt.show()

    elif (which_to_run == "3"):
        best_r2_val = -np.inf
        best_forest = None
        best_params = None

        for max_feat in range(1, len(da.features) + 1):
            for min_leaf in [1, 10]:
                forest = RandomForestRegressor(n_estimators=100, max_features=max_feat, min_samples_leaf=min_leaf, random_state=1, n_jobs=-1)
                forest.fit(X, y)

                r2_val = r2_score(y_val, forest.predict(X_val))
                print(f"max_features = {max_feat}, min_samples_leaf = {min_leaf}: validation R2 = {r2_val:.4f}")

                if r2_val > best_r2_val:
                    best_r2_val = r2_val
                    best_forest = forest
                    best_params = (max_feat, min_leaf)

        print(f"\nBest: max_features = {best_params[0]}, min_samples_leaf = {best_params[1]}")
        print("R2 of random forest on training: ", r2_score(y, best_forest.predict(X)))
        print("R2 of random forest on validation: ", best_r2_val)
        y_random_forest_pred = best_forest.predict(X_test)
        r2_test = r2_score(y_test, y_random_forest_pred)
        print("R2 of random forest on test: ", r2_test)
        print("MAPE of random forest on test: ", mean_absolute_percentage_error(y_test, y_random_forest_pred), "\n")

        for name, importance in sorted(zip(da.features, best_forest.feature_importances_), key=lambda p: -p[1]):
            print(f"  {name}: {importance:.3f}")

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlabel("random forest prediction (MWh)")
        ax.set_ylabel("actual consumption (MWh)")
        ax.set_title(f"Random forest, test R2 = {r2_test:.2f}")
        ax.scatter(y_random_forest_pred, y_test, s=2, alpha=0.2, c="skyblue", label="test datapoints")
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='r', linewidth=2, label="prediction = actual")
        ax.legend()
        plt.show()

