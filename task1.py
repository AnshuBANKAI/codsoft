import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder 
from sklearn.impute import SimpleImputer 
from sklearn.ensemble import RandomForestClassifier 
  


data = pd.read_csv("Titanic-Dataset.csv")

# Let’s see what the data looks like
print("Here’s the first few rows of our data:")
print(data.head())

# chacking data details!!
print("\n My Data Details:")
print(data.info())

# Searching missing value in data.
print("\n Detail of missing data?")
print(data.isnull().sum())

# Creating plot screen.
plt.figure(figsize=(12, 8))

#creating Sub plot screen for each data.
#plot1
plt.subplot(2, 2, 1) 
sns.countplot(x='Sex', hue='Survived', data=data)
plt.title('Who Survived by Gender?')

#plot2
plt.subplot(2, 2, 2) 
sns.countplot(x='Pclass', hue='Survived', data=data)
plt.title('Survival by Ticket Class')

#plot3
plt.subplot(2, 2, 3) 
sns.histplot(data=data, x='Age', hue='Survived', kde=True) 
plt.title('Age vs. Survival')

#plot4
plt.subplot(2, 2, 4)  
number_columns = ['Age', 'SibSp', 'Parch', 'Fare']
sns.heatmap(data[number_columns].corr(), annot=True, cmap='coolwarm')
plt.title('How Numbers Relate')
plt.tight_layout()
plt.show()

#There position 
data = data.drop(columns=['Cabin'])

# Some ages are missing.
age_filler = SimpleImputer(strategy='median')
data['Age'] = age_filler.fit_transform(data[['Age']])

#Their boarding port.
port_filler = SimpleImputer(strategy='most_frequent')
data['Embarked'] = port_filler.fit_transform(data[['Embarked']]).ravel()

# Turn ‘Gender’ into numbers: male = 1, female = 0
sex_encoder = LabelEncoder()
data['Sex'] = sex_encoder.fit_transform(data['Sex'])

#like checkboxes
data = pd.get_dummies(data, columns=['Embarked', 'Pclass'], drop_first=True)

#Predicating ticket 
data = data.drop(columns=['PassengerId', 'Name', 'Ticket'])

#Per person detail by creating method 
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

data = data.drop(columns=['SibSp', 'Parch'])

# Now split the data: X is everything except ‘Survived’, y is ‘Survived’
X = data.drop(columns=['Survived'])
y = data['Survived']

# Use 80% of the data to train, 20% to test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model (it’s like asking 100 iron to vote on who survives)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)




# What factors mattered most for survival?
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nWhat Mattered Most for Survival?")
print(importance)

# Plot the important factors
plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=importance)
plt.title('My Model study')
plt.show()