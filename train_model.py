"""
ML Model Training Pipeline for Drug Side Effect Risk Prediction
Trains XGBoost, Random Forest, and Logistic Regression classifiers.
Saves the best model artifacts for the Flask web application.
"""

import os
import sys
import json
import joblib

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'drug_side_effect_dataset.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(MODELS_DIR, exist_ok=True)


def train_and_save():
    """Full training pipeline."""
    print("=" * 60)
    print("  Drug Side Effect Risk Prediction — Model Training")
    print("=" * 60)
    
    # 1. Load dataset
    print("\n[1/6] Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"    Dataset shape: {df.shape}")
    print(f"    Columns: {list(df.columns)}")
    print(f"    Target distribution:\n{df['Side_Effect_Risk'].value_counts().to_string()}")
    
    # 2. Feature engineering
    print("\n[2/6] Preparing features...")
    
    # Drop internal score column (label leakage)
    feature_cols = [c for c in df.columns if c not in ['Risk_Score_Internal', 'Side_Effect_Risk']]
    X = df[feature_cols].copy()
    y = df['Side_Effect_Risk'].copy()
    
    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    print(f"    Target classes: {list(target_encoder.classes_)}")
    
    # Encode categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
    print(f"    Categorical features: {categorical_cols}")
    print(f"    Numerical features: {numerical_cols}")
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Save feature names
    feature_names = list(X.columns)
    
    # 3. Train/test split
    print("\n[3/6] Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"    Train: {X_train.shape[0]} samples")
    print(f"    Test: {X_test.shape[0]} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Train multiple models
    print("\n[4/6] Training models...")
    
    models = {
        'XGBoost': XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n  Training {name}...")
        
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=target_encoder.classes_, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        results[name] = {
            'accuracy': round(acc * 100, 2),
            'cv_mean': round(cv_scores.mean() * 100, 2),
            'cv_std': round(cv_scores.std() * 100, 2),
            'classification_report': report,
            'confusion_matrix': cm
        }
        
        print(f"    Accuracy: {acc*100:.2f}%")
        print(f"    CV Score: {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")
        print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))
    
    # 5. Select best model
    print("\n[5/6] Selecting best model...")
    best_name = max(results, key=lambda k: results[k]['accuracy'])
    best_model = models[best_name]
    print(f"    [OK] Best model: {best_name} ({results[best_name]['accuracy']}%)")
    
    # 6. Save artifacts
    print("\n[6/6] Saving model artifacts...")
    
    # Save model
    model_path = os.path.join(MODELS_DIR, 'xgb_side_effect_model.joblib')
    joblib.dump(best_model, model_path)
    print(f"    Saved: {model_path}")
    
    # Save encoders
    encoders_path = os.path.join(MODELS_DIR, 'label_encoders.joblib')
    joblib.dump(label_encoders, encoders_path)
    print(f"    Saved: {encoders_path}")
    
    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, 'feature_scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"    Saved: {scaler_path}")
    
    # Save target encoder
    target_enc_path = os.path.join(MODELS_DIR, 'target_encoder.joblib')
    joblib.dump(target_encoder, target_enc_path)
    print(f"    Saved: {target_enc_path}")
    
    # Save feature names
    feat_path = os.path.join(MODELS_DIR, 'feature_names.json')
    with open(feat_path, 'w', encoding='utf-8') as f:
        json.dump(feature_names, f, indent=2)
    print(f"    Saved: {feat_path}")
    
    # Save metadata
    metadata = {
        'best_model': best_name,
        'accuracy': results[best_name]['accuracy'],
        'cv_mean': results[best_name]['cv_mean'],
        'cv_std': results[best_name]['cv_std'],
        'version': '1.0',
        'features_count': len(feature_names),
        'training_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'target_classes': list(target_encoder.classes_)
    }
    meta_path = os.path.join(MODELS_DIR, 'best_model_metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"    Saved: {meta_path}")
    
    # Save full comparison results
    comparison_path = os.path.join(MODELS_DIR, 'model_comparison_results.json')
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"    Saved: {comparison_path}")
    
    print("\n" + "=" * 60)
    print("  Training Complete! All artifacts saved to ./models/")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    train_and_save()
