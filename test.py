from pymongo import MongoClient
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


client = MongoClient("mongodb://localhost:27017/")
db = client["ecommerce_db"]
collection = db["sales"]

sample_data = [
    {"date": "2023-01-01", "product": "Laptop", "sales": 10},
    {"date": "2023-01-02", "product": "Laptop", "sales": 12},
    {"date": "2023-01-03", "product": "Laptop", "sales": 15},
    {"date": "2023-01-04", "product": "Laptop", "sales": 9},
    {"date": "2023-01-05", "product": "Laptop", "sales": 18}
]
collection.insert_many(sample_data)
print("Sample data inserted.")

sales_data = list(collection.find())
df = pd.DataFrame(sales_data)

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date')

df['day'] = (df['date'] - df['date'].min()).dt.days
print("Data fetched and prepared:\n", df)

X = df['day'].values.reshape(-1, 1)
y = df['sales'].values

model = LinearRegression()
model.fit(X, y)

future_days = np.arange(df['day'].max() + 1, df['day'].max() + 11).reshape(-1, 1)
predictions = model.predict(future_days)

for i, pred in enumerate(predictions, start=1):
    print(f"Day {i}: Predicted Sales = {pred:.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Sales')
plt.plot(future_days, predictions, color='red', label='Predicted Sales')
plt.xlabel('Days')
plt.ylabel('Sales')
plt.title('Actual vs Predicted Sales')
plt.legend()
plt.show()

predicted_sales = [
    {"date": (df['date'].max() + pd.Timedelta(days=i)).strftime('%Y-%m-%d'),
     "product": "Laptop", "predicted_sales": int(pred)}
    for i, pred in enumerate(predictions, start=1)
]
collection.insert_many(predicted_sales)
print("Predictions saved to MongoDB.")
