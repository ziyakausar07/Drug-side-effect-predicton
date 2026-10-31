# Drug Side Effect Prediction & Drug Information System Using Machine Learning (PharmacoRisk AI)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask 3.0+](https://img.shields.io/badge/Framework-Flask_3.0-green.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-red.svg)](https://xgboost.readthedocs.io/)
[![OpenCV](https://img.shields.io/badge/Vision-OpenCV_OCR-informational.svg)](https://opencv.org/)
[![License: Academic](https://img.shields.io/badge/License-Academic_Research_Only-yellow.svg)](#important-safety-notice--disclaimer)

---

## Important Safety Notice & Disclaimer
> **Academic / Research Demonstration Notice:**  
> This application is developed strictly for **educational and research purposes**. Machine Learning predictions are statistical estimates and are **not medical diagnoses, clinical prescriptions, or treatment recommendations**. Do not start, stop, or change medication based on this application. Always consult a qualified healthcare professional (physician or clinical pharmacist) for medical advice.

---

## 1. Project Title
**Drug Side Effect Prediction and Drug Information System Using Machine Learning**

---

## 2. Introduction
Adverse Drug Reactions (ADRs) are among the leading causes of morbidity and preventable hospital admissions worldwide. Predicting a patient's individual susceptibility to drug side effects requires evaluating multifaceted physiological factors—including age, renal biomarkers (serum creatinine), hepatic transaminases (ALT), baseline vitals, known allergen profiles, and concomitant medications.

**PharmacoRisk AI** is an end-to-end Machine Learning and Computer Vision web application designed to assess potential adverse side-effect risk (**Low, Moderate, High**) and deliver structured pharmacological drug monographs through text search or OCR packaging recognition.

---

## 3. Problem Statement
Traditional pharmacological prescribing often relies on general population guidelines that cannot dynamically integrate complex patient-specific polypharmacy, renal/hepatic biomarkers, and previous adverse reaction history. Furthermore, patients and students often lack intuitive, accessible systems to verify medication safety warnings and understand individual vulnerability factors.

---

## 4. Project Objectives
1. **Develop an ML Multi-Class Classification Pipeline** to predict adverse side-effect risk (`Low`, `Moderate`, `High`).
2. **Benchmark 5 Classification Algorithms** (Logistic Regression, Decision Tree, Random Forest, XGBoost, Support Vector Machine) with comprehensive metrics (Accuracy, Macro Precision, Macro Recall, Macro F1, and Multi-class ROC-AUC).
3. **Implement Computer Vision & OCR** (OpenCV preprocessing + Tesseract OCR with fuzzy string matching) to identify medication names from packaging/blister photos.
4. **Provide Clinical Explainability** by attributing risk predictions to patient-specific physiological biomarkers and pharmacological interactions.
5. **Deliver a Production-Ready Web Application** with patient clinical intake forms, safety monographs, administrative audit dashboards, and REST APIs.

---

## 5. Key Features
- **Patient Clinical Intake Form**: Validated entry of age, gender, weight, blood pressure, blood glucose, serum creatinine, ALT liver enzyme, chronic comorbidities, allergy history, and concomitant medications.
- **Dual Drug Identification**:
  - **Catalog Search**: Live dropdown and autocomplete search across verified pharmacological monographs.
  - **Image Scan (OCR)**: OpenCV contrast enhancement, adaptive thresholding, and Tesseract OCR with fallback confidence scoring.
- **Multi-Class ML Risk Estimation**: Delivers class predictions alongside probability distributions (e.g., Low 15%, Moderate 65%, High 20%).
- **Clinical Explainability**: Generates clear, human-understandable clinical rationale factors based on feature weights and biomarker thresholds.
- **Comprehensive Drug Monograph Library**: Displays indications, common side effects, black-box warnings, contraindications, drug interactions, and precautions across 14,000+ pharmacological records.
- **Interactive Analytics Dashboard**: Live Chart.js telemetry tracking risk distribution, algorithm comparison, and prediction history.
- **RESTful API Endpoints**: Programmatic interfaces for drug search, OCR packaging recognition, and patient risk inference.

---

## 6. Technology Stack
- **Backend**: Python 3.10–3.13, Flask 3.0, SQLite3, Jinja2
- **Machine Learning**: Scikit-Learn, XGBoost, Pandas, NumPy, Joblib
- **Image Processing & OCR**: OpenCV (cv2), PyTesseract, Pillow
- **Frontend**: HTML5, CSS3, JavaScript (ES6+), Bootstrap 5.3, FontAwesome 6, Chart.js
- **Testing & Quality Assurance**: Automated Pytest suite (15 test cases)

---

## 7. System Architecture
```
                         +-----------------------------------+
                         |          Web Client (UI)          |
                         |  (Bootstrap 5, JS, Chart.js)      |
                         +-----------------+-----------------+
                                           | HTTP Requests
                                           v
                         +-----------------------------------+
                         |         Flask Application         |
                         |             (app.py)              |
                         +--------+------------------+-------+
                                  |                  |
           +----------------------+                  +-----------------------+
           |                                                                 |
           v                                                                 v
+-----------------------+                                         +---------------------+
|  Image Processing &   |                                         |  ML Inference &     |
|  OCR Engine (utils/)  |                                         |  Explainability     |
|  - OpenCV Thresholding|                                         |  - Scalers/Encoders |
|  - Tesseract OCR      |                                         |  - XGBoost Model    |
|  - Fuzzy DB Matcher   |                                         |  - Clinical Rules   |
+----------+------------+                                         +----------+----------+
           |                                                                 |
           +----------------------+                  +-----------------------+
                                  |                  |
                                  v                  v
                         +-----------------------------------+
                         |      SQLite Database (db.db)      |
                         |  (Drugs, Patients, Predictions,   |
                         |   Admin Users, OCR Logs)          |
                         +-----------------------------------+
```

---

## 8. Machine Learning Methodology
1. **Dataset Generation & Clinical Realism**: 6,000 synthetic patient-medication records modeling realistic pharmacokinetic and pharmacodynamic interactions (geriatric clearance decay, renal contraindications, drug-drug enzyme competition, allergen cross-reactivity).
2. **Preprocessing Pipeline**:
   - Numerical Scaling: `StandardScaler` for continuous features (`Patient_Age`, `Weight_kg`, `Systolic_BP`, `Diastolic_BP`, `Blood_Sugar_mg_dL`, `Creatinine_mg_dL`, `ALT_Enzyme_U_L`).
   - Categorical Encoding: `OneHotEncoder(handle_unknown='ignore')` across multi-category features.
   - Target Encoding: Multi-class mapped labels (`Low: 0`, `Moderate: 1`, `High: 2`).
3. **Model Training & Comparison**: Stratified 80/20 train/test split evaluated across 5 standard algorithms with balanced class weighting.

---

## 9. Model Benchmark Comparison
Evaluated on **1,200 held-out stratified test cases**:

| Algorithm | Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro F1-Score (%) | Multi-Class ROC-AUC (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** *(Selected)* | **85.33%** | **78.25%** | **66.75%** | **70.94%** | **93.56%** |
| Support Vector Machine (RBF) | 79.75% | 68.10% | 71.57% | 69.40% | 91.46% |
| Random Forest Classifier | 78.25% | 65.80% | 71.01% | 67.62% | 91.03% |
| Decision Tree Classifier | 77.00% | 60.10% | 65.83% | 60.72% | 79.47% |
| Logistic Regression | 74.33% | 58.40% | 69.97% | 62.43% | 87.63% |

*Decision Rule: XGBoost was selected as the final production model due to superior multi-class discrimination (ROC-AUC 93.56%) and highest macro F1-score (70.94%).*

---

## 10. Project Directory Structure
```text
drug-side-effect-prediction/
│
├── app.py                             # Main Flask application & API routes
├── requirements.txt                   # Python dependency specifications
├── README.md                          # Project overview and run guide
├── PROJECT_REPORT.md                  # Comprehensive academic report
├── Procfile                           # Cloud deployment configuration
├── train_model.py                     # ML model training & evaluation script
├── update_conditions_and_train.py     # Condition mapping & retraining script
├── import_dataset.py                  # Clinical dataset import helper
├── share_online.py                    # Local tunneling / share helper
│
├── dataset/
│   └── drug_side_effect_dataset.csv   # Clinical dataset records
│
├── models/
│   ├── xgb_side_effect_model.joblib   # Serialized production XGBoost model
│   ├── feature_scaler.joblib          # StandardScaler for numerical biomarkers
│   ├── feature_names.json             # Feature names ordering
│   ├── label_encoders.joblib          # Categorical label encoders
│   ├── target_encoder.joblib          # Risk level target encoder
│   ├── best_model_metadata.json       # Model configuration & performance metrics
│   └── model_comparison_results.json  # Multi-algorithm benchmark data
│
├── database/
│   ├── __init__.py
│   ├── schema.sql                     # Relational SQLite schema
│   ├── db.py                          # Thread-safe database helper functions
│   ├── seed_data.py                   # Drug catalog seeder
│   ├── conditions.json                # Pre-configured medical conditions list
│   └── database.db                    # SQLite persistent database (14,400+ drugs)
│
├── static/
│   └── uploads/                       # User packaging upload directory
│
├── templates/
│   ├── base.html                      # Layout base template (Bootstrap 5.3 dark theme)
│   ├── index.html                     # Landing page with workflow & catalog explorer
│   ├── patient.html                   # Clinical patient intake form with demo presets
│   ├── drug_upload.html               # Drug selection & OCR image scanning
│   ├── prediction.html                # Prediction risk badges, probabilities & explainability
│   ├── drug_info.html                 # Drug monographs & safety reference
│   └── dashboard.html                 # Analytics telemetry & model audit
│
├── utils/
│   ├── __init__.py
│   ├── image_processor.py             # OpenCV image thresholding & validation
│   ├── ocr.py                         # Tesseract OCR & fuzzy string matching
│   └── prediction.py                  # ML inference & clinical rule attribution
│
└── tests/
    └── test_app.py                    # 15 automated pytest tests (routes, ML, validation)
```

---

## 11. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed
- Git installed
- Optional: Tesseract OCR binary (for physical packaging OCR scanning)

### Step 1: Clone Repository
```powershell
git clone https://github.com/ziyakausar07/Drug-side-effect-predicton.git
cd Drug-side-effect-predicton
```

### Step 2: Install Python Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run Database Initialization & Seeding (Pre-seeded in database.db)
```powershell
python database/seed_data.py
```

### Step 4: Run Machine Learning Training Pipeline (Optional / Pre-Trained)
```powershell
python train_model.py
```

### Step 5: Execute Automated Test Suite
```powershell
pytest tests/
```

---

## 12. How to Run the Web Application
Start the Flask development server:
```powershell
python app.py
```

Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 13. Step-by-Step Viva Demonstration Walkthrough
1. **Landing Page (`/`)**: Walk through the system concept and explore the pre-loaded educational drugs catalog.
2. **Patient Form (`/patient`)**: Use the **"Quick Demo Presets"** dropdown:
   - Select *Geriatric Patient with CKD* (Age: 74, Creatinine: 2.4 mg/dL).
   - Click **Continue to Drug Selection**.
3. **Drug Selection (`/drug-upload`)**:
   - Choose a contraindicated drug (e.g., *Metformin* or *Lisinopril*) with *High Dose*.
   - Alternatively, upload a medicine packaging image and click **Extract & Identify Drug**.
4. **Prediction Result (`/prediction/<id>`)**:
   - Highlight the **High Risk Badge** and probability breakdown.
   - Point out the **Explainability Section**: explains why the combination was flagged (elevated creatinine, geriatric clearance, dosage intensity).
5. **Analytics Dashboard (`/dashboard`)**:
   - Inspect live risk distribution doughnut chart and the 5-algorithm ML benchmark grouped bar chart.
   - Review prediction telemetry and model configuration metrics.

---

## 14. Academic Limitations
1. **Synthetic Data**: The training dataset is synthetically generated according to established pharmacokinetic rules for educational demonstration; it does not replace real-world clinical trial databases.
2. **OCR Variance**: Image text extraction depends on lighting and label resolution. When confidence is below 68%, the system requires manual user selection to prevent clinical errors.
3. **Statistical Nature**: Machine learning models calculate mathematical associations rather than deterministic biochemical outcomes.

---

## 15. Future Scope
- Integration with live Electronic Health Record (EHR) systems via FHIR APIs.
- Pharmacogenomics integration (CYP2D6, CYP2C19 genetic polymorphism markers).
- Deep learning computer vision for pill shape and imprint recognition (CNN / Vision Transformers).
- Multi-drug regimen matrix prediction (polypharmacy interaction graphs).

---

## 16. Academic Viva Voce Questions & Answers
- **Q: Why was XGBoost chosen over Logistic Regression?**  
  *A:* XGBoost handles complex non-linear feature interactions (e.g., age combined with creatinine and drug class) and achieved the highest ROC-AUC (93.56%) and Macro F1 (70.94%).
- **Q: How is class imbalance handled?**  
  *A:* Through balanced class weighting (`class_weight='balanced'`) during loss optimization and stratified splitting to maintain class proportions.
- **Q: How does the OCR prevent false medication identification?**  
  *A:* A fuzzy string matching threshold (68% token ratio) is enforced. If confidence is insufficient, the system explicitly prompts the user to select manually.
