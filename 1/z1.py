import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Učitavanje podataka iz CSV fajlova
train_data = pd.read_csv('../titanic/train.csv')
test_data = pd.read_csv('../titanic/test.csv')

# Popunjavanje nedostajućih vrednosti
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

# Kombinovanje trening i test podataka
combined_data = pd.concat([train_data[['Sex', 'Embarked']], test_data[['Sex', 'Embarked']]])

# Kodiranje kategorijskih promenljivih
label_encoder = LabelEncoder()
combined_data['Sex'] = label_encoder.fit_transform(combined_data['Sex'])
combined_data['Embarked'] = label_encoder.fit_transform(combined_data['Embarked'])

# Vraćamo kodirane vrednosti nazad u odgovarajuće DataFrame-ove
train_data['Sex'] = combined_data['Sex'][:len(train_data)]
train_data['Embarked'] = combined_data['Embarked'][:len(train_data)]

test_data['Sex'] = combined_data['Sex'][len(train_data):]
test_data['Embarked'] = combined_data['Embarked'][len(train_data):]

# Kreiranje nove funkcije 'FamilySize' kao zbir 'SibSp' i 'Parch'
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch']
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch']

# Priprema podataka za obuku modela
X = train_data[['Pclass', 'Sex', 'Age', 'FamilySize', 'Fare', 'Embarked']]
y = train_data['Survived']

# Stablo odlučivanja
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Predikcija za test podatke
test_predictions = model.predict(test_data[['Pclass', 'Sex', 'Age', 'FamilySize', 'Fare', 'Embarked']])

# Prikazivanje rezultata predikcija
test_data['Survived'] = test_predictions

# Ispis tačnosti modela na trening i test skupu
train_accuracy = model.score(X, y)
print(f'Tačnost modela na trening skupu: {train_accuracy * 100:.2f}%')

# Spremanje rezultata u CSV fajl
test_data[['PassengerId', 'Survived']].to_csv('titanic_predictions.csv', index=False)



