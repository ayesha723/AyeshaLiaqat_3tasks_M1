# -----------------------------------
# California Housing Price Prediction
# -----------------------------------

# 📦 Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🔧 Preprocessing & model tools
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

# 📊 Statsmodels for VIF (optional advanced diagnostics)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# 🏠 Read dataset
df = pd.read_csv(r"f:\artificial intelligence\project4\housing.csv")
print(df.head())

# ----------------------------
# 🧹 Basic data exploration
# ----------------------------
print(f"Shape of dataset: {df.shape}")
print("\nMissing values per column:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nData types:\n", df.dtypes)

# 🔥 Correlation heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# 🛠 Fill missing values
df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)

# 🔍 Re-check for missing values
print("\nMissing values after fill:\n", df.isnull().sum())

# --------------------------------------
# 📊 Distributions of numeric features
# --------------------------------------
fig, ax = plt.subplots(4, 2, figsize=(14, 12))
cols = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
        'total_bedrooms', 'population', 'households', 'median_house_value']
for i, col in enumerate(cols):
    sns.histplot(df[col], kde=True, ax=ax[i//2, i%2])
plt.tight_layout()
plt.show()

# 📉 Categorical vs Target
sns.barplot(x='ocean_proximity', y='median_house_value', data=df, palette="Set1")
plt.title("House Value by Ocean Proximity")
plt.xticks(rotation=45)
plt.show()

# 🌍 House value across geographic coordinates
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='longitude', y='latitude', hue='median_house_value', palette='coolwarm', alpha=0.5)
plt.title("Geographic Distribution of Median House Value")
plt.show()

# 🧠 Label encode 'ocean_proximity'
le = LabelEncoder()
df['ocean_proximity'] = le.fit_transform(df['ocean_proximity'])

# 🔁 Re-check Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="Pastel1")
plt.title("Correlation Matrix After Encoding")
plt.show()

# --------------------------------------
# 🏗 Split data into features & target
# --------------------------------------
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# 🧪 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# 🧼 Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------------
# 📊 Fit Linear, Lasso & Ridge Models
# -------------------------------------
# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Lasso Regression
lasso = Lasso()
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

# Ridge Regression
ridge = Ridge()
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

# -------------------------------------
# 📏 Metrics Function
# -------------------------------------
def get_metrics(p, y, y_pred):
    n = len(y)
    r2 = r2_score(y, y_pred)
    adjusted_r2 = 1 - (((1 - r2) * (n - 1)) / (n - p - 1))
    mae = mean_absolute_error(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    return r2, adjusted_r2, mae, mape, mse

# -------------------------------------------------
# (c) 📉 Visualize model predictions vs actual prices
# -------------------------------------------------
figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

# Linear Regression
df_lr = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).reset_index(drop=True)
axes[0].plot(df_lr[:20])
axes[0].set_title("Linear Regression")
axes[0].legend(["Actual", "Predicted"], loc="upper left")

# Lasso Regression
df_lasso = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_lasso}).reset_index(drop=True)
axes[1].plot(df_lasso[:20])
axes[1].set_title("Lasso Regression")
axes[1].legend(["Actual", "Predicted"], loc="upper left")

# Ridge Regression
df_ridge = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_ridge}).reset_index(drop=True)
axes[2].plot(df_ridge[:20])
axes[2].set_title("Ridge Regression")
axes[2].legend(["Actual", "Predicted"], loc="upper left")

plt.tight_layout()
plt.show()

# ----------------------------------------
# 📊 Compare Model Performance Metrics
# ----------------------------------------
p = X_train.shape[1]  # Number of features
performance_df = pd.DataFrame([
    get_metrics(p, y_test, y_pred),
    get_metrics(p, y_test, y_pred_lasso),
    get_metrics(p, y_test, y_pred_ridge)
], 
columns=['R2', 'Adjusted R2', 'MAE', 'MAPE', 'MSE'],
index=['Linear Regression', 'Lasso Regression', 'Ridge Regression'])

print("\n📈 Model Performance Comparison:")
print(performance_df)
