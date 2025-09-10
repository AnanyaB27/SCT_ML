import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

# Feature engineering: Bathrooms = FullBath + HalfBath * 0.5
train["Bathrooms"] = train["FullBath"] + 0.5 * train["HalfBath"]
test["Bathrooms"] = test["FullBath"] + 0.5 * test["HalfBath"]

# Select features
X = train[["GrLivArea", "BedroomAbvGr", "Bathrooms"]]
y = train["SalePrice"]
X_test_final = test[["GrLivArea", "BedroomAbvGr", "Bathrooms"]]

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Validation check
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse}")

# Predict on test data
test_predictions = model.predict(X_test_final)

# Save predictions
submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": test_predictions
})
submission.to_csv("my_submission.csv", index=False)

print("✅ Predictions saved to my_submission.csv")

"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Feature engineering
train["Bathrooms"] = train["FullBath"] + 0.5 * train["HalfBath"]
test["Bathrooms"] = test["FullBath"] + 0.5 * test["HalfBath"]

# Choose more features (based on domain knowledge + Kaggle solutions)
features = [
    "GrLivArea",      # Above ground living area
    "BedroomAbvGr",   # Bedrooms
    "Bathrooms",      # Bathrooms
    "OverallQual",    # Overall quality (very strong feature)
    "GarageCars",     # Size of garage in cars
    "GarageArea",     # Garage size in sqft
    "TotalBsmtSF",    # Basement area
    "1stFlrSF",       # First floor area
    "YearBuilt",      # Year built
    "YearRemodAdd"    # Year remodeled
]

# Fill missing values with median (simple strategy)
for col in features:
    train[col] = train[col].fillna(train[col].median())
    test[col] = test[col].fillna(train[col].median())

X = train[features]
y = train["SalePrice"]
X_test_final = test[features]

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Validation
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.2f}")

# Predict on test data
test_predictions = model.predict(X_test_final)

# Save predictions
submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": test_predictions
})
submission.to_csv("my_submission.csv", index=False)

print("✅ Predictions saved to my_submission.csv")
"""