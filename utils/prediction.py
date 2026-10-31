"""
ML Prediction Engine for Drug Side Effect Risk Assessment
Loads pre-trained XGBoost model artifacts and provides risk prediction.
Falls back to a rule-based heuristic when ML artifacts are not yet trained.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

# Paths to model artifacts
_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
_MODEL_PATH = os.path.join(_MODELS_DIR, 'xgb_side_effect_model.joblib')
_ENCODERS_PATH = os.path.join(_MODELS_DIR, 'label_encoders.joblib')
_SCALER_PATH = os.path.join(_MODELS_DIR, 'feature_scaler.joblib')
_METADATA_PATH = os.path.join(_MODELS_DIR, 'best_model_metadata.json')
_FEATURE_NAMES_PATH = os.path.join(_MODELS_DIR, 'feature_names.json')

# In-memory cache
_artifacts = {
    'model': None,
    'encoders': None,
    'scaler': None,
    'metadata': None,
    'feature_names': None,
    'loaded': False
}

# Risk class mapping
RISK_CLASSES = ['High', 'Low', 'Moderate']


def load_artifacts():
    """Pre-load model artifacts into memory for fast inference."""
    if not os.path.exists(_MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {_MODEL_PATH}. "
            "Run 'python train_model.py' first to train the ML model."
        )
    
    _artifacts['model'] = joblib.load(_MODEL_PATH)
    _artifacts['encoders'] = joblib.load(_ENCODERS_PATH) if os.path.exists(_ENCODERS_PATH) else {}
    _artifacts['scaler'] = joblib.load(_SCALER_PATH) if os.path.exists(_SCALER_PATH) else None
    _artifacts['metadata'] = {}
    _artifacts['feature_names'] = []
    
    if os.path.exists(_METADATA_PATH):
        with open(_METADATA_PATH, 'r', encoding='utf-8') as f:
            _artifacts['metadata'] = json.load(f)
    
    if os.path.exists(_FEATURE_NAMES_PATH):
        with open(_FEATURE_NAMES_PATH, 'r', encoding='utf-8') as f:
            _artifacts['feature_names'] = json.load(f)
    
    _artifacts['loaded'] = True


def _encode_categorical(value, encoder_key):
    """Encode a categorical value using the saved LabelEncoder."""
    encoders = _artifacts.get('encoders', {})
    if encoder_key in encoders:
        le = encoders[encoder_key]
        try:
            return le.transform([value])[0]
        except ValueError:
            # Unknown category — return the most common class (0)
            return 0
    return 0


def _build_feature_vector(patient_data, drug_data, dosage_category):
    """
    Build a numeric feature vector matching the training schema.
    
    Features (matching dataset columns):
    - Patient_Age, Gender, Weight_kg, Medical_Condition, Allergy_History,
      Previous_ADR, Current_Medications, Systolic_BP, Diastolic_BP,
      Blood_Sugar_mg_dL, Creatinine_mg_dL, ALT_Enzyme_U_L
    - Drug_Name, Drug_Category, Dosage_Category
    """
    if isinstance(drug_data, str):
        drug_data = {'drug_name': drug_data, 'category': ''}
    elif not isinstance(drug_data, dict):
        drug_data = {'drug_name': getattr(drug_data, 'drug_name', 'Unknown'), 'category': getattr(drug_data, 'category', 'Unknown')} if drug_data else {}

    features = {
        'Patient_Age': patient_data.get('age', 30),
        'Gender': patient_data.get('gender', 'Male'),
        'Weight_kg': patient_data.get('weight', 70.0),
        'Medical_Condition': patient_data.get('medical_condition', 'None'),
        'Allergy_History': patient_data.get('allergy_history', 'None'),
        'Previous_ADR': patient_data.get('previous_adr', 'No'),
        'Current_Medications': patient_data.get('current_medication', 'None'),
        'Systolic_BP': patient_data.get('systolic_bp', 120),
        'Diastolic_BP': patient_data.get('diastolic_bp', 80),
        'Blood_Sugar_mg_dL': patient_data.get('blood_sugar', 100.0),
        'Creatinine_mg_dL': patient_data.get('creatinine', 1.0),
        'ALT_Enzyme_U_L': patient_data.get('alt_enzyme', 25.0),
        'Drug_Name': drug_data.get('drug_name', 'Unknown'),
        'Drug_Category': drug_data.get('category', 'Unknown'),
        'Dosage_Category': dosage_category
    }
    
    # Encode categorical features
    categorical_cols = ['Gender', 'Medical_Condition', 'Allergy_History', 
                       'Previous_ADR', 'Current_Medications', 'Drug_Name', 
                       'Drug_Category', 'Dosage_Category']
    
    feature_vector = []
    feature_names = _artifacts.get('feature_names', [])
    
    if feature_names:
        for fname in feature_names:
            if fname in features:
                val = features[fname]
                if fname in categorical_cols:
                    val = _encode_categorical(val, fname)
                feature_vector.append(float(val))
            else:
                feature_vector.append(0.0)
    else:
        # Fallback ordering
        ordered_keys = ['Patient_Age', 'Gender', 'Weight_kg', 'Medical_Condition',
                       'Allergy_History', 'Previous_ADR', 'Current_Medications',
                       'Systolic_BP', 'Diastolic_BP', 'Blood_Sugar_mg_dL',
                       'Creatinine_mg_dL', 'ALT_Enzyme_U_L', 'Drug_Name',
                       'Drug_Category', 'Dosage_Category']
        for key in ordered_keys:
            val = features.get(key, 0)
            if key in categorical_cols:
                val = _encode_categorical(val, key)
            feature_vector.append(float(val))
    
    fnames = _artifacts.get('feature_names')
    if fnames and len(fnames) == len(feature_vector):
        return pd.DataFrame([feature_vector], columns=fnames)
    return np.array([feature_vector])


def _identify_contributing_factors(patient_data, drug_data, dosage_category):
    """Rule-based identification of key clinical risk factors."""
    factors = []
    
    if isinstance(drug_data, str):
        drug_data = {'drug_name': drug_data, 'category': ''}
    elif not isinstance(drug_data, dict):
        drug_data = {'drug_name': getattr(drug_data, 'drug_name', ''), 'category': getattr(drug_data, 'category', '')} if drug_data else {}

    age = patient_data.get('age', 30)
    creatinine = patient_data.get('creatinine', 1.0)
    alt = patient_data.get('alt_enzyme', 25.0)
    blood_sugar = patient_data.get('blood_sugar', 100.0)
    systolic_bp = patient_data.get('systolic_bp', 120)
    allergy = patient_data.get('allergy_history', 'None')
    previous_adr = patient_data.get('previous_adr', 'No')
    condition = patient_data.get('medical_condition', 'None')
    drug_name = drug_data.get('drug_name', '')
    drug_category = drug_data.get('category', '')
    
    # Age-related
    if age >= 65:
        factors.append(f"Elderly patient (Age: {age}) — increased pharmacokinetic sensitivity")
    elif age <= 12:
        factors.append(f"Pediatric patient (Age: {age}) — dosage adjustment may be required")
    
    # Renal function
    if creatinine > 1.5:
        factors.append(f"Elevated serum creatinine ({creatinine} mg/dL) — potential renal impairment")
    
    # Liver function
    if alt > 40:
        factors.append(f"Elevated ALT enzyme ({alt} U/L) — potential hepatic concern")
    
    # Blood sugar
    if blood_sugar > 200:
        factors.append(f"Elevated blood glucose ({blood_sugar} mg/dL) — hyperglycemia risk")
    
    # Blood pressure
    if systolic_bp >= 160:
        factors.append(f"Severe hypertension (Systolic: {systolic_bp} mmHg)")
    elif systolic_bp >= 140:
        factors.append(f"Stage 1 hypertension (Systolic: {systolic_bp} mmHg)")
    
    # Allergy cross-check
    if allergy and allergy != 'None':
        if drug_category and allergy.lower() in drug_category.lower():
            factors.append(f"⚠ Known allergy to '{allergy}' — drug is in the same class!")
        elif allergy.lower() in drug_name.lower():
            factors.append(f"⚠ Known allergy specifically to '{drug_name}'!")
        else:
            factors.append(f"Allergy history: {allergy}")
    
    # Previous ADR
    if previous_adr == 'Yes':
        factors.append("History of prior Adverse Drug Reactions (ADR)")
    
    # Medical conditions affecting drug risk
    if condition and condition != 'None':
        factors.append(f"Pre-existing condition: {condition}")
    
    # Dosage
    if dosage_category == 'High':
        factors.append("High dosage category — increased risk of dose-dependent adverse effects")
    
    if not factors:
        factors.append("No significant clinical risk factors identified")
    
    return factors


def predict_side_effect_risk(patient_data, drug_data, dosage_category='Standard'):
    """
    Main prediction function.
    
    Args:
        patient_data: dict with patient clinical parameters
        drug_data: dict or string with drug information
        dosage_category: 'Low', 'Standard', or 'High'
    
    Returns:
        dict with predicted_risk, probabilities, contributing_factors
    """
    if not _artifacts.get('loaded'):
        load_artifacts()

    if isinstance(drug_data, str):
        drug_data = {'drug_name': drug_data, 'category': ''}
    elif not isinstance(drug_data, dict):
        drug_data = {'drug_name': getattr(drug_data, 'drug_name', 'Unknown'), 'category': getattr(drug_data, 'category', 'Unknown')} if drug_data else {}

    contributing_factors = _identify_contributing_factors(patient_data, drug_data, dosage_category)
    
    # Try ML model first
    if _artifacts.get('loaded') and _artifacts['model'] is not None:
        try:
            X = _build_feature_vector(patient_data, drug_data, dosage_category)
            
            # Scale if scaler is available
            if _artifacts['scaler'] is not None:
                X = _artifacts['scaler'].transform(X)
            
            model = _artifacts['model']
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            
            # Map to risk classes
            predicted_risk = RISK_CLASSES[int(prediction)]
            prob_dict = {
                RISK_CLASSES[i]: round(float(probabilities[i]), 4)
                for i in range(len(RISK_CLASSES))
            }
            
            return {
                'predicted_risk': predicted_risk,
                'probabilities': prob_dict,
                'contributing_factors': contributing_factors,
                'model_used': 'XGBoost Classifier',
                'model_version': _artifacts.get('metadata', {}).get('version', '1.0')
            }
            
        except Exception as e:
            print(f"[!] ML prediction error, falling back to heuristic: {e}")
    
    # Fallback: rule-based heuristic
    return _heuristic_prediction(patient_data, drug_data, dosage_category, contributing_factors)


def _heuristic_prediction(patient_data, drug_data, dosage_category, factors):
    """
    Rule-based fallback prediction when ML model is not available.
    Uses clinical parameters to estimate risk.
    """
    risk_score = 0.0
    
    age = patient_data.get('age', 30)
    creatinine = patient_data.get('creatinine', 1.0)
    alt = patient_data.get('alt_enzyme', 25.0)
    blood_sugar = patient_data.get('blood_sugar', 100.0)
    systolic_bp = patient_data.get('systolic_bp', 120)
    allergy = patient_data.get('allergy_history', 'None')
    previous_adr = patient_data.get('previous_adr', 'No')
    condition = patient_data.get('medical_condition', 'None')
    drug_category = drug_data.get('category', '')
    
    # Age contribution
    if age >= 65:
        risk_score += 2.0
    elif age >= 50:
        risk_score += 1.0
    elif age <= 12:
        risk_score += 1.5
    
    # Renal
    if creatinine > 2.0:
        risk_score += 3.0
    elif creatinine > 1.5:
        risk_score += 2.0
    
    # Hepatic
    if alt > 60:
        risk_score += 2.5
    elif alt > 40:
        risk_score += 1.5
    
    # Cardiovascular
    if systolic_bp >= 160:
        risk_score += 2.0
    elif systolic_bp >= 140:
        risk_score += 1.0
    
    # Glycemic
    if blood_sugar > 250:
        risk_score += 2.0
    elif blood_sugar > 180:
        risk_score += 1.0
    
    # Allergy match
    if allergy and allergy != 'None':
        if drug_category and allergy.lower() in drug_category.lower():
            risk_score += 4.0
        else:
            risk_score += 1.0
    
    # Prior ADR
    if previous_adr == 'Yes':
        risk_score += 2.0
    
    # Medical condition
    if condition and condition != 'None':
        risk_score += 1.0
    
    # Dosage
    if dosage_category == 'High':
        risk_score += 2.0
    elif dosage_category == 'Low':
        risk_score -= 1.0
    
    # Classify
    if risk_score >= 7.0:
        predicted_risk = 'High'
        prob_high = min(0.95, 0.5 + risk_score * 0.04)
        prob_mod = (1 - prob_high) * 0.7
        prob_low = 1 - prob_high - prob_mod
    elif risk_score >= 3.5:
        predicted_risk = 'Moderate'
        prob_mod = min(0.85, 0.4 + risk_score * 0.05)
        prob_high = (1 - prob_mod) * 0.4
        prob_low = 1 - prob_mod - prob_high
    else:
        predicted_risk = 'Low'
        prob_low = min(0.95, 0.6 + (3.5 - risk_score) * 0.05)
        prob_mod = (1 - prob_low) * 0.7
        prob_high = 1 - prob_low - prob_mod
    
    return {
        'predicted_risk': predicted_risk,
        'probabilities': {
            'Low': round(max(0, prob_low), 4),
            'Moderate': round(max(0, prob_mod), 4),
            'High': round(max(0, prob_high), 4)
        },
        'contributing_factors': factors,
        'model_used': 'Clinical Heuristic (Rule-Based Fallback)',
        'model_version': 'heuristic-v1'
    }
