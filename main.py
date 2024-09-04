import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
df = pd.read_csv(r'D:\Desktop\Python\Weather Forcasting\weatherHistory.csv')

df = df.head(100)
label_encoder = LabelEncoder()
df['Summary'] = label_encoder.fit_transform(df['Summary'])
df['Precip Type'] = label_encoder.fit_transform(df['Precip Type'].astype(str))
X = df.drop(['Formatted Date', 'Daily Summary', 'Temperature (C)'], axis=1)  # Use Temperature (C) as the target
y = df['Temperature (C)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Temperature')
plt.plot(y_pred, label='Predicted Temperature', linestyle='--')
plt.title('Actual vs Predicted Temperature (First 50 Data Points)')
plt.xlabel('Sample Index')
plt.ylabel('Temperature (C)')
plt.legend()
plt.show()

example_input = X_test.iloc[0]
predicted_temperature = model.predict([example_input])
print(f'Predicted Temperature: {predicted_temperature[0]}')
