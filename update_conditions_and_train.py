"""
Update Medical Conditions and Train ML Pipeline on Comprehensive Dataset
"""

import os
import sys
import json
import sqlite3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'database', 'database.db')
CONDITIONS_JSON_PATH = os.path.join(BASE_DIR, 'database', 'conditions.json')
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'drug_side_effect_dataset.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Update Medical Conditions
# -------------------------------------------------------------
def update_medical_conditions():
    print("[1/4] Compiling medical conditions list...")
    standards = [
        "None",
        "Acne",
        "ADHD (Attention Deficit Hyperactivity Disorder)",
        "AIDS / HIV Infection",
        "Alcohol Withdrawal / Dependence",
        "Allergies / Allergic Rhinitis",
        "Alzheimer's Disease",
        "Amyotrophic Lateral Sclerosis (ALS)",
        "Anemia (Iron Deficiency / Chronic Disease)",
        "Angina Pectoris (Chest Pain)",
        "Anxiety Disorder (Generalized / Social)",
        "Arrhythmia / Atrial Fibrillation",
        "Arthritis (Osteoarthritis / Rheumatoid)",
        "Asthma (Bronchial)",
        "Atherosclerosis",
        "Autism Spectrum Disorder",
        "Bacterial Infections (Respiratory / Skin / UTI)",
        "Benign Prostatic Hyperplasia (BPH)",
        "Bipolar Disorder (Mania / Depression)",
        "Bladder Infection / Cystitis",
        "Bronchitis (Acute & Chronic)",
        "Cancer / Oncology Chemotherapy",
        "Cardiovascular Disease",
        "Celiac Disease",
        "Chronic Fatigue Syndrome",
        "Chronic Kidney Disease (CKD)",
        "Chronic Obstructive Pulmonary Disease (COPD)",
        "Chronic Pain Syndrome",
        "Cirrhosis / Chronic Liver Disease",
        "Colds & Flu (Viral Upper Respiratory)",
        "Colitis / Ulcerative Colitis",
        "Constipation (Chronic / Acute)",
        "Coronary Artery Disease",
        "Covid-19 Infection",
        "Crohn's Disease",
        "Deep Vein Thrombosis (DVT)",
        "Dementia / Cognitive Impairment",
        "Depression (Major Depressive Disorder)",
        "Dermatitis (Contact / Atopic / Seborrheic)",
        "Diabetes Mellitus (Type 1)",
        "Diabetes Mellitus (Type 2)",
        "Diabetic Neuropathy / Retinopathy",
        "Diarrhea / Acute Gastroenteritis",
        "Dry Eye Syndrome",
        "Dyslipidemia / Hypercholesterolemia",
        "Eczema",
        "Emergency Contraception",
        "Endometriosis",
        "Epilepsy / Seizure Disorder",
        "Erectile Dysfunction",
        "Fibromyalgia",
        "Fungal Infections (Candida / Tinea)",
        "Gastroesophageal Reflux Disease (GERD / Acid Reflux)",
        "Glaucoma / Ocular Hypertension",
        "Gout / Hyperuricemia",
        "Hair Loss / Alopecia",
        "Heart Failure (Congestive Heart Failure)",
        "Hemorrhoids / Piles",
        "Hepatitis (Viral Hepatitis B / C)",
        "Herpes Simplex / Herpes Zoster (Shingles)",
        "High Blood Pressure / Essential Hypertension",
        "Hyperlipidemia / Elevated Triglycerides",
        "Hypertension",
        "Hyperthyroidism / Graves' Disease",
        "Hypokalemia / Electrolyte Imbalance",
        "Hypothyroidism / Hashimoto's",
        "Inflammatory Bowel Disease (IBD)",
        "Insomnia / Sleep Disorders",
        "Irritable Bowel Syndrome (IBS)",
        "Kidney Stones (Nephrolithiasis)",
        "Leukemia",
        "Lupus (Systemic Lupus Erythematosus)",
        "Lymphoma",
        "Macular Degeneration",
        "Malaria",
        "Menopause Symptoms / Vasomotor Hot Flashes",
        "Migraine Headache (Acute & Prophylaxis)",
        "Multiple Sclerosis",
        "Muscle Spasms / Musculoskeletal Pain",
        "Myocardial Infarction (Post-Heart Attack)",
        "Nausea & Vomiting",
        "Neuropathic Pain (Neuralgia)",
        "Neutropenia",
        "Non-Alcoholic Fatty Liver Disease (NAFLD)",
        "Obesity / Weight Management",
        "Obsessive Compulsive Disorder (OCD)",
        "Ocular Inflammation / Conjunctivitis",
        "Osteoarthritis",
        "Osteopenia / Bone Mineral Loss",
        "Osteoporosis",
        "Otitis Media (Middle Ear Infection)",
        "Pain Management (Acute / Postoperative / Chronic)",
        "Panic Disorder / Agoraphobia",
        "Parkinson's Disease",
        "Peptic Ulcer Disease / Gastric Ulcers",
        "Peripheral Artery Disease",
        "Pharyngitis / Strep Throat",
        "Pneumonia (Bacterial / Viral)",
        "Polycystic Ovary Syndrome (PCOS)",
        "Post-Traumatic Stress Disorder (PTSD)",
        "Postoperative Pain / Inflammation",
        "Pre-Eclampsia",
        "Prostate Cancer",
        "Psoriasis / Plaque Psoriasis",
        "Psoriatic Arthritis",
        "Pulmonary Embolism",
        "Pulmonary Fibrosis",
        "Pulmonary Hypertension",
        "Restless Legs Syndrome",
        "Rheumatoid Arthritis",
        "Rosacea",
        "Schizophrenia / Schizoaffective Disorder",
        "Sciatica / Lumbar Radiculopathy",
        "Seasonal Allergies (Hay Fever)",
        "Sinusitis (Acute / Chronic)",
        "Skin Infections (Cellulitis / Impetigo)",
        "Social Anxiety Disorder",
        "Stomach Ulcers / Acid Dyspepsia",
        "Stroke / Transient Ischemic Attack (TIA)",
        "Tachycardia / Palpitations",
        "Tendinitis / Bursitis",
        "Thrombocytopenia",
        "Thyroiditis",
        "Tinnitus",
        "Tonsillitis",
        "Tourette Syndrome",
        "Trigeminal Neuralgia",
        "Tuberculosis",
        "Ulcerative Colitis",
        "Urinary Incontinence",
        "Urinary Tract Infection (UTI)",
        "Urticaria / Hives",
        "Vaginal Candidiasis / Yeast Infection",
        "Varicose Veins / Venous Insufficiency",
        "Vertigo / Meniere's Disease",
        "Viral Infections",
        "Vitamin D / B12 Deficiency",
        "Vitiligo",
        "Weight Loss / Cachexia"
    ]

    cond_set = set(standards)

    # Also load from drugs.com dataset if available
    csv_drugs_com = r"E:\Project\Drug Side Effects\Drug_side_effect\drugs_side_effects_drugs_com.csv"
    if os.path.exists(csv_drugs_com):
        df_dc = pd.read_csv(csv_drugs_com, usecols=['medical_condition'])
        for c in df_dc['medical_condition'].dropna().unique():
            c_str = str(c).strip()
            if 2 < len(c_str) < 50:
                cond_set.add(c_str)

    sorted_conditions = sorted(list(cond_set), key=lambda x: (x != "None", x.lower()))

    # Update database
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS medical_conditions (id INTEGER PRIMARY KEY AUTOINCREMENT, condition_name TEXT UNIQUE)")
    cur.execute("DELETE FROM medical_conditions")
    for cond in sorted_conditions:
        cur.execute("INSERT OR IGNORE INTO medical_conditions (condition_name) VALUES (?)", (cond,))
    conn.commit()
    conn.close()

    # Save to JSON
    with open(CONDITIONS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(sorted_conditions, f, indent=2)

    print(f"    Saved {len(sorted_conditions)} medical conditions into SQLite & conditions.json.")
    return sorted_conditions


# -------------------------------------------------------------
# 2. Generate Augmented Training Dataset Covering Full Drug Catalog
# -------------------------------------------------------------
def build_expanded_training_dataset():
    print("\n[2/4] Building expanded multi-drug clinical training dataset...")
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT drug_name, category FROM drugs")
    drug_rows = cur.fetchall()
    conn.close()

    print(f"    Found {len(drug_rows)} drugs in database.")

    # Select representative drug sample from database (150+ diverse drugs)
    np.random.seed(42)
    sample_indices = np.random.choice(len(drug_rows), size=min(len(drug_rows), 200), replace=False)
    selected_drugs = [drug_rows[i] for i in sample_indices]
    
    # Always include classic reference drugs
    core_drugs = [
        ("Metformin", "Antidiabetic (Biguanide)"),
        ("Lisinopril", "ACE Inhibitor (Antihypertensive)"),
        ("Amoxicillin", "Antibiotic (Aminopenicillin)"),
        ("Atorvastatin", "Statin (HMG-CoA Reductase Inhibitor)"),
        ("Ibuprofen", "NSAID (Nonsteroidal Anti-inflammatory)"),
        ("Omeprazole", "Proton Pump Inhibitor (PPI)"),
        ("Paracetamol", "Analgesic / Antipyretic"),
        ("Ciprofloxacin", "Fluoroquinolone Antibiotic"),
        ("Azithromycin", "Macrolide Antibiotic"),
        ("Warfarin", "Anticoagulant (Vitamin K Antagonist)"),
        ("Sertraline", "SSRI Antidepressant"),
        ("Metoprolol", "Beta-Blocker"),
        ("Prednisone", "Corticosteroid"),
        ("Amlodipine", "Calcium Channel Blocker"),
        ("Levothyroxine", "Thyroid Hormone")
    ]
    all_drugs_pool = core_drugs + [d for d in selected_drugs if d[0] not in [c[0] for c in core_drugs]]

    with open(CONDITIONS_JSON_PATH, 'r', encoding='utf-8') as f:
        conditions_list = json.load(f)

    allergies_list = [
        'None', 'Penicillin', 'NSAIDs / Aspirin', 'Sulfonamides', 
        'ACE Inhibitors', 'Statins', 'Opioids', 'Macrolides', 'Fluoroquinolones'
    ]
    concomitant_meds = [
        'None', 'Metformin', 'Lisinopril', 'ACE Inhibitors', 'Beta-Blockers', 
        'Diuretics', 'SSRIs', 'PPIs', 'Statins', 'Anticoagulants', 'Warfarin', 'Aspirin'
    ]
    dosage_levels = ['Low', 'Standard', 'High']

    # Generate 10,000 diverse clinical patient records
    records = []
    n_samples = 10000

    for i in range(n_samples):
        age = int(np.random.choice([
            np.random.randint(18, 45),
            np.random.randint(45, 65),
            np.random.randint(65, 88)
        ], p=[0.35, 0.40, 0.25]))

        gender = np.random.choice(['Male', 'Female'])
        weight = round(float(np.random.normal(74, 15)), 1)
        weight = max(42.0, min(140.0, weight))

        cond = np.random.choice(conditions_list)
        allergy = np.random.choice(allergies_list, p=[0.60, 0.08, 0.08, 0.06, 0.05, 0.04, 0.03, 0.03, 0.03])
        prev_adr = np.random.choice(['No', 'Yes'], p=[0.82, 0.18])
        current_med = np.random.choice(concomitant_meds, p=[0.35] + [0.65/11]*11)

        # Baseline vitals and lab values
        sys_bp = int(np.random.normal(125 if age < 50 else 138, 16))
        sys_bp = max(90, min(210, sys_bp))
        dia_bp = int(sys_bp * 0.65 + np.random.normal(0, 5))
        dia_bp = max(55, min(120, dia_bp))

        glucose = round(float(np.random.normal(105 if 'Diabetes' not in cond else 165, 28)), 1)
        glucose = max(65.0, min(380.0, glucose))

        creatinine = round(float(np.random.normal(1.0 if 'Kidney' not in cond else 1.9, 0.35)), 2)
        creatinine = max(0.5, min(5.5, creatinine))

        alt = round(float(np.random.normal(25 if 'Liver' not in cond else 55, 12)), 1)
        alt = max(10.0, min(180.0, alt))

        drug_item = all_drugs_pool[np.random.randint(0, len(all_drugs_pool))]
        drug_name = drug_item[0]
        drug_cat = drug_item[1]
        dosage = np.random.choice(dosage_levels, p=[0.25, 0.55, 0.20])

        # Clinical Risk Engine
        risk_score = 0.0
        if age >= 65: risk_score += 2.0
        elif age >= 50: risk_score += 1.0

        if creatinine > 1.8: risk_score += 3.2
        elif creatinine > 1.3: risk_score += 1.8

        if alt > 50: risk_score += 2.5
        elif alt > 38: risk_score += 1.2

        if sys_bp >= 160: risk_score += 2.0
        elif sys_bp >= 140: risk_score += 1.0

        if glucose > 220: risk_score += 2.0
        elif glucose > 160: risk_score += 1.0

        if prev_adr == 'Yes': risk_score += 2.0

        # Allergy match
        if allergy != 'None':
            if allergy.lower() in drug_cat.lower() or allergy.lower() in drug_name.lower():
                risk_score += 5.0
            else:
                risk_score += 0.8

        if cond != 'None':
            risk_score += 1.0

        if dosage == 'High': risk_score += 2.2
        elif dosage == 'Low': risk_score -= 1.0

        # Interaction check
        if current_med != 'None':
            if ('Anticoagulant' in drug_cat or 'Warfarin' in drug_name) and ('NSAID' in drug_cat or 'Aspirin' in current_med):
                risk_score += 4.0
            elif ('ACE' in drug_cat) and ('NSAID' in current_med or 'Potassium' in current_med):
                risk_score += 2.5
            else:
                risk_score += 0.8

        # Classify risk
        if risk_score >= 6.8:
            risk_class = 'High'
        elif risk_score >= 3.2:
            risk_class = 'Moderate'
        else:
            risk_class = 'Low'

        records.append({
            'Patient_Age': age,
            'Gender': gender,
            'Weight_kg': weight,
            'Medical_Condition': cond,
            'Allergy_History': allergy,
            'Previous_ADR': prev_adr,
            'Current_Medications': current_med,
            'Systolic_BP': sys_bp,
            'Diastolic_BP': dia_bp,
            'Blood_Sugar_mg_dL': glucose,
            'Creatinine_mg_dL': creatinine,
            'ALT_Enzyme_U_L': alt,
            'Drug_Name': drug_name,
            'Drug_Category': drug_cat,
            'Dosage_Category': dosage,
            'Risk_Score_Internal': round(risk_score, 2),
            'Side_Effect_Risk': risk_class
        })

    df = pd.DataFrame(records)
    df.to_csv(DATASET_PATH, index=False)
    print(f"    Saved augmented dataset to {DATASET_PATH} with {len(df)} rows.")
    print(f"    Risk class distribution:\n{df['Side_Effect_Risk'].value_counts().to_string()}")
    return df


# -------------------------------------------------------------
# 3. Train Models and Save Artifacts
# -------------------------------------------------------------
def train_and_evaluate(df):
    print("\n[3/4] Training multi-model classification pipeline...")
    
    # Feature columns (exclude internal score and target)
    feature_cols = [c for c in df.columns if c not in ['Risk_Score_Internal', 'Side_Effect_Risk']]
    X = df[feature_cols].copy()
    y = df['Side_Effect_Risk'].copy()

    # Target encoder
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    joblib.dump(target_encoder, os.path.join(MODELS_DIR, 'target_encoder.joblib'))

    # Encode categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    joblib.dump(label_encoders, os.path.join(MODELS_DIR, 'label_encoders.joblib'))

    # Save feature names
    feature_names = list(X.columns)
    with open(os.path.join(MODELS_DIR, 'feature_names.json'), 'w', encoding='utf-8') as f:
        json.dump(feature_names, f, indent=2)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Standard Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'feature_scaler.joblib'))

    # Initialize models
    models = {
        'XGBoost': XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            eval_metric='mlogloss'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    }

    comparison = {}
    best_model_name = None
    best_acc = 0.0
    best_model_obj = None

    for name, model in models.items():
        print(f"    Training {name}...")
        # Train on scaled features
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        
        # 5-fold CV
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())

        print(f"      -> Test Accuracy: {acc*100:.2f}% | 5-Fold CV: {cv_mean*100:.2f}% (+/- {cv_std*100:.2f}%)")

        comparison[name] = {
            'accuracy': round(acc * 100, 2),
            'cv_mean': round(cv_mean * 100, 2),
            'cv_std': round(cv_std * 100, 2)
        }

        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_model_obj = model

    # Save best model
    best_model_path = os.path.join(MODELS_DIR, 'xgb_side_effect_model.joblib')
    joblib.dump(best_model_obj, best_model_path)
    print(f"\n[+] Saved best model ({best_model_name}) to {best_model_path}")

    # Save metadata
    metadata = {
        'best_model': best_model_name,
        'accuracy': round(best_acc * 100, 2),
        'cv_mean': comparison[best_model_name]['cv_mean'],
        'cv_std': comparison[best_model_name]['cv_std'],
        'version': '2.0-comprehensive',
        'features_count': len(feature_names),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'target_classes': list(target_encoder.classes_)
    }
    with open(os.path.join(MODELS_DIR, 'best_model_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(MODELS_DIR, 'model_comparison_results.json'), 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)

    print("[4/4] Model training and evaluation successfully completed!")
    print(f"      Best Model: {best_model_name} ({metadata['accuracy']}% Accuracy)")


if __name__ == "__main__":
    update_medical_conditions()
    df = build_expanded_training_dataset()
    train_and_evaluate(df)
