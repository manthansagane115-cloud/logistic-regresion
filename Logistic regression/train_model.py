import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load training data
train_df = pd.read_csv('Titanic_train.csv')

# Handle missing values
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

# Drop unnecessary columns
df = train_df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1)

# Encode categorical variables
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])  # male: 1, female: 0
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Prepare data
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split and train
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the model
with open('titanic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully as 'titanic_model.pkl'")
print(f"Model accuracy on validation set: {model.score(X_val, y_val):.4f}")
