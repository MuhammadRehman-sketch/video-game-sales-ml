# Import all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# --- Step 1: Load the Data ---
df = pd.read_csv('Video Games Sales (1980-2024) - Raw.csv')

# --- Step 2: Data Cleaning ---
# Drop rows with no sales data and fill missing critic scores with the average
df = df.dropna(subset=['total_sales'])
df['critic_score'] = df['critic_score'].fillna(df['critic_score'].mean())
df = df.drop(columns=['img', 'last_update'])

# --- Step 3: Exploratory Data Analysis (EDA) ---
# Calculate and plot total sales by genre
genre_sales = df.groupby('genre')['total_sales'].sum().sort_values(ascending=False)
genre_sales.plot(kind='bar', color='skyblue', figsize=(10,5))
plt.title('Total Sales by Genre')
plt.ylabel('Sales (Millions)')
plt.show()

# --- Step 4: Feature Engineering (Encoding) ---
# Convert text categories into numbers
le = LabelEncoder()
df['genre_n'] = le.fit_transform(df['genre'])
df['console_n'] = le.fit_transform(df['console'])

# Set up the Features (X) and Target (y)
X = df[['genre_n', 'console_n', 'critic_score']]
y = df['total_sales']

# --- Step 5: Training the Model ---
# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest algorithm
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Step 6: Evaluation ---
# Test the model and calculate the error
predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)

# Output the final result
print("-" * 30)
print(f"Average Error in Sales Prediction: ${round(error, 2)} Million")
print("-" * 30)