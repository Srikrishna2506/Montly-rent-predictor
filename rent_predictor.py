import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Updated data
data = {
    'Month': list(range(1, 13)),
    'Jaipur': [1100, 1120, 1140, 1160, 1175, 1190, 1200, 1220, 1240, 1260, 1280, 1300],
    'Delhi': [1600, 1620, 1650, 1680, 1700, 1720, 1750, 1780, 1800, 1820, 1850, 1880],
    'Vijayawada': [900, 910, 920, 930, 940, 950, 960, 970, 980, 990, 1000, 1010],
    'Mumbai': [2000, 2020, 2040, 2050, 2070, 2090, 2100, 2120, 2150, 2170, 2190, 2200],
    'Hyderabad': [1300, 1320, 1340, 1350, 1365, 1380, 1395, 1400, 1420, 1435, 1450, 1470],
}

df = pd.DataFrame(data)
df['Average_Rent'] = df[['Jaipur', 'Delhi', 'Vijayawada', 'Mumbai', 'Hyderabad']].mean(axis=1)

# Preprocessing
X = df[['Month']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = df['Average_Rent']

# Linear Regression Model
model = LinearRegression()
model.fit(X_scaled, y)

# Prediction for Month 13
X_13_scaled = scaler.transform([[13]])
predicted_avg_rent = model.predict(X_13_scaled)[0]
y_pred = model.predict(X_scaled)

# Evaluation
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Predicted Average Rent for Month 13: ${predicted_avg_rent:.2f}")
print(f"\nModel Evaluation Metrics: \nMSE: {mse:.2f}, R²: {r2:.2f}")

# Bar plot for each month and city
df_melted = df.melt(id_vars='Month', value_vars=['Jaipur', 'Delhi', 'Vijayawada', 'Mumbai', 'Hyderabad'],
                    var_name='City', value_name='Rent')

plt.figure(figsize=(12, 6))
sns.barplot(data=df_melted, x='Month', y='Rent', hue='City')
plt.title("Monthly Rent Comparison Across Cities")
plt.xlabel("Month")
plt.ylabel("Rent Price ($)")
plt.legend(title='City')
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()