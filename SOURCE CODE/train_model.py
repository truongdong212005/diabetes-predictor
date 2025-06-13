from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import os

# load dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# load dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

print(f'Dataset shape: {X.shape}')
print(f'Feature names: {diabetes.feature_names}')
print(f'Target range: {y.min():.1f} to {y.max():.1f}')

# train model
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# evaluate model
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}')
print(f'R2 Score: {r2_score(y_test, y_pred):.3f}')

# save trained model to reuse next time
os.makedirs('models', exist_ok=True)

with open('models/diabetes_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print('\n--Train model and save successfully--')