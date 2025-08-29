import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib


data = pd.read_csv('pcos_data.csv')

X = data[['Age', 'BMI', 'FSH', 'LH', 'CycleIrregular', 'HairGrowth', 'Acne']]
y = data['PCOS']


model = RandomForestClassifier()
model.fit(X, y)


joblib.dump(model, 'pcos_model.pkl')

print("Model trained and saved as pcos_model.pkl")
