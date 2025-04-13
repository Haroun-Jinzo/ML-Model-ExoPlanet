import numpy as np
import pandas as pd
from scipy.fft import fft
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
import joblib
import matplotlib.pyplot as plt

# ----------------------
# 1. Feature Engineering
# ----------------------

def extract_features(flux_data):
    """Convert raw flux time-series to engineered features"""
    features = []
    for series in flux_data:
        mean = np.mean(series)
        std = np.std(series)
        slope = np.polyfit(range(len(series)), series, 1)[0]
        fft_vals = np.abs(fft(series))
        dominant_freq = np.argmax(fft_vals[1:len(fft_vals)//2]) + 1
        spectral_entropy = -np.sum(fft_vals * np.log(fft_vals + 1e-12))
        features.append([mean, std, slope, dominant_freq, spectral_entropy])
    return np.array(features)

# ----------------------
# 2. Model Training
# ----------------------

def train_models(X_train_raw, y_train, X_test_raw, y_test):
    # Convert labels and verify classes
    y_train = np.where(y_train == 2, 1, 0)
    y_test = np.where(y_test == 2, 1, 0)
    
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training data contains only one class")
    
    # Feature engineering
    X_train = extract_features(X_train_raw)
    X_test = extract_features(X_test_raw)
    
    # Handle NaNs/Infs
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)
    
    # Feature scaling (critical for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create validation split
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.1, stratify=y_train
    )
    
    # ----------------------
    # XGBoost Training
    # ----------------------
    print("\nTraining XGBoost...")
    xgb_model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        early_stopping_rounds=10,
        scale_pos_weight=np.sum(y_train == 0)/np.sum(y_train == 1),
        n_jobs=-1
    )
    
    xgb_params = {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
    xgb_grid = GridSearchCV(xgb_model, xgb_params, scoring='precision', cv=3, verbose=2)
    xgb_grid.fit(X_train_final, y_train_final, eval_set=[(X_val, y_val)], verbose=False)
    
    # ----------------------
    # SVM Training
    # ----------------------
    print("\nTraining SVM...")
    svm_model = SVC(
        class_weight='balanced',
        probability=True,  # Required for ROC AUC
        random_state=42
    )
    
    svm_params = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear']
    }
    
    svm_grid = GridSearchCV(svm_model, svm_params, scoring='precision', cv=3, verbose=2)
    svm_grid.fit(X_train_scaled, y_train)
    
    # ----------------------
    # Random Forest Training
    # ----------------------
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Save models
    joblib.dump(xgb_grid.best_estimator_, 'xgb_model.pkl')
    joblib.dump(svm_grid.best_estimator_, 'svm_model.pkl')
    joblib.dump(rf_model, 'rf_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    # Evaluate
    print("\nXGBoost Evaluation:")
    evaluate_model(xgb_grid.best_estimator_, X_test_scaled, y_test)
    
    print("\nSVM Evaluation:")
    evaluate_model(svm_grid.best_estimator_, X_test_scaled, y_test)
    
    print("\nRandom Forest Evaluation:")
    evaluate_model(rf_model, X_test_scaled, y_test)
    
    return xgb_grid.best_estimator_, svm_grid.best_estimator_, rf_model, scaler

# ----------------------
# 3. Evaluation
# ----------------------

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    print(classification_report(y_test, y_pred, target_names=['No Planet', 'Planet']))
    if y_proba is not None:
        print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    
    # Feature importance for tree-based models
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        feat_importances = pd.Series(model.feature_importances_,
                                   index=['Mean', 'Std', 'Slope',
                                         'Dominant Freq', 'Spectral Entropy'])
        feat_importances.nlargest(5).plot(kind='barh')
        plt.title(f'{model.__class__.__name__} Feature Importance')
        plt.show()

# ----------------------
# 4. Main Execution
# ----------------------

if __name__ == "__main__":
   # Load your pre-split data here
    train_df = pd.read_csv('exoTrain.csv')
    test_df = pd.read_csv('exoTest.csv')

    X_train_raw = train_df.drop('LABEL', axis=1).values
    y_train = train_df['LABEL'].values                   

    X_test_raw = test_df.drop('LABEL', axis=1).values  
    y_test = test_df['LABEL'].values
    # Train models
    xgb_model, svm_model, rf_model, scaler = train_models(
        X_train_raw, y_train, X_test_raw, y_test
    )