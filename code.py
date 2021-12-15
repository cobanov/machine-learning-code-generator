
# ==================
# Please provide data

data = pd.read_csv()
# ==================

# Importings
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Model Pre-processing
data.dropna()
pd.get_dummies()

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1)

# Model Fitting
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Prediction
preds = model.predict(X_test)

# Evaluate
score = mean_absolute_error(preds, y_test)
print(score)

