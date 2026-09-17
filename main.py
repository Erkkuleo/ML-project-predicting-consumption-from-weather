import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_percentage_error



df = pd.read_csv('data/train.csv')

df.drop(columns=['time_utc', 'local_time', 'is_weekend', 'hour'], inplace=True)

features = ['temp_c', "wind_ms", "humid_pc", "hour_sin", "hour_cos", "weekday", "month"]

X = df[features].to_numpy()
y = df['consumption_mwh'].to_numpy()

poly = PolynomialFeatures(degree= 3)
X_poly = poly.fit_transform(X)

lin_regr = LinearRegression(fit_intercept=False)
lin_regr.fit(X_poly, y)

y_pred = lin_regr.predict(X_poly)
r2 = r2_score(y,y_pred)
print("R2 of polynomial regression : ", r2)


fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel("feature value, scaled to 0-1 (0 = feature min, 1 = feature max)")
ax.set_ylabel("consumption (MWh)")
ax.set_title("Consumption vs all features")
colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink']

for i, name in enumerate(features):
    x_min, x_max = X[:,i].min(), X[:,i].max()
    ax.scatter((X[:,i] - x_min) / (x_max - x_min), y, s=1, alpha=0.03, color=colors[i])
    X_fit = np.linspace(x_min, x_max, 100)
    x_grid = np.tile(X.mean(axis=0), (100,1))
    x_grid[:,i] = X_fit
    ax.plot((X_fit - x_min) / (x_max - x_min), lin_regr.predict(poly.transform(x_grid)), color=colors[i], linewidth=2, label=name)

ax.legend(title="model prediction (other features at mean)")
plt.show()