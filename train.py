import os
import json
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = "dataset/winequality-red.csv"
df = pd.read_csv(DATA_PATH, sep=";")


os.makedirs("results", exist_ok=True)

X = df.drop("quality", axis=1)
y = df["quality"]


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


joblib.dump(model, "results/model.joblib")


metrics = {
    "r2": float(r2),
    "mse": float(mse)
}


with open("results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Final R2 Score: {r2}")
print(f"Final MSE: {mse}")