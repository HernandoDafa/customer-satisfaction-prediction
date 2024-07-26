import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib

# Memuat dataset
df = pd.read_csv('onlinefoods.csv')

# Menghapus baris dengan nilai yang hilang
df.dropna(inplace=True)

# Mendefinisikan fitur dan target
X = df.drop('Output', axis=1)
y = df['Output']

# Mengidentifikasi kolom kategorikal dan numerik
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(include=['number']).columns

# One-hot encoding untuk kolom kategorikal
encoder = OneHotEncoder(drop='first', sparse=False)
encoded_categorical = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))
encoded_categorical.columns = encoder.get_feature_names_out(categorical_cols)
encoded_categorical.index = X.index

# Menggabungkan kembali data setelah encoding
X = X.drop(categorical_cols, axis=1)
X = pd.concat([X, encoded_categorical], axis=1)

# Standard scaling untuk fitur numerik
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning untuk Logistic Regression
param_grid_log_reg = {'C': [0.01, 0.1, 1, 10, 100]}
grid_log_reg = GridSearchCV(LogisticRegression(), param_grid_log_reg, cv=5)
grid_log_reg.fit(X_train, y_train)
best_log_reg = grid_log_reg.best_estimator_

# Hyperparameter Tuning untuk Decision Tree
param_grid_dec_tree = {'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10]}
grid_dec_tree = GridSearchCV(DecisionTreeClassifier(), param_grid_dec_tree, cv=5)
grid_dec_tree.fit(X_train, y_train)
best_dec_tree = grid_dec_tree.best_estimator_

# Hyperparameter Tuning untuk K-Nearest Neighbors
param_grid_knn = {'n_neighbors': [3, 5, 7, 10], 'weights': ['uniform', 'distance']}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5)
grid_knn.fit(X_train, y_train)
best_knn = grid_knn.best_estimator_

# Memilih model terbaik berdasarkan hasil tuning
# Misalnya kita memilih Logistic Regression sebagai model terbaik
best_model = best_log_reg

# Simpan model terbaik
joblib.dump(best_model, 'best_model.pkl')
