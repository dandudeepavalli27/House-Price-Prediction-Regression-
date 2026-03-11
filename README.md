# 1 Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
# 2 Load Dataset
df = pd.read_csv("House Price Prediction Dataset.csv")
print(df.head())
print(df.info())
# 3 Exploratory Data Analysis
sns.histplot(df["Price"], kde=True)
plt.title("House Price Distribution")
plt.show()
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
# 4 Feature Engineering
df["Price"] = np.log1p(df["Price"])
current_year = 2025
df["HouseAge"] = current_year - df["YearBuilt"]
df = pd.get_dummies(df, drop_first=True)
# 5 Features and Target
X = df.drop(["Price","Id"], axis=1)
y = df["Price"]
# 6 Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
# 7 Evaluation Function
def evaluate(y_true, pred):
    rmse = np.sqrt(mean_squared_error(y_true, pred))
    mae = mean_absolute_error(y_true, pred)
    r2 = r2_score(y_true, pred)
    print("RMSE:", rmse)
    print("MAE :", mae)
    print("R2  :", r2)
    print("------------------")
# 8 Linear Regression
print("Linear Regression")
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
evaluate(y_test, pred_lr)
# 9 Ridge Regression
print("Ridge Regression")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
pred_ridge = ridge.predict(X_test)
evaluate(y_test, pred_ridge)
# 10 Lasso Regression
print("Lasso Regression")
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
pred_lasso = lasso.predict(X_test)
evaluate(y_test, pred_lasso)
# 11 XGBoost Model
print("XGBoost")
xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5
)

xgb.fit(X_train, y_train)
pred_xgb = xgb.predict(X_test)
evaluate(y_test, pred_xgb)
# 12 Stacking Model
print("Stacking Model")
estimators = [
    ("ridge", Ridge()),
    ("lasso", Lasso()),
    ("xgb", XGBRegressor())
]

stack = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression()
)

stack.fit(X_train, y_train)
pred_stack = stack.predict(X_test)
evaluate(y_test, pred_stack)
# 13 Feature Importance
importance = pd.Series(
    xgb.feature_importances_,
    index=X.columns
)

importance.sort_values().plot(
    kind="barh",
    title="Feature Importance"
)
plt.show()
