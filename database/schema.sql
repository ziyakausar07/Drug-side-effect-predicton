-- Database Schema for Drug Side Effect Prediction and Information System

CREATE TABLE IF NOT EXISTS drugs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    drug_name TEXT NOT NULL UNIQUE,
    generic_name TEXT NOT NULL,
    category TEXT NOT NULL,
    uses TEXT NOT NULL,
    common_side_effects TEXT NOT NULL,
    serious_warnings TEXT NOT NULL,
    contraindications TEXT NOT NULL,
    interactions TEXT NOT NULL,
    precautions TEXT NOT NULL,
    source_reference TEXT DEFAULT 'FDA / DailyMed / WHO Essential Medicines',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER NOT NULL,
    gender TEXT NOT NULL,
    weight REAL NOT NULL,
    medical_condition TEXT NOT NULL,
    allergy_history TEXT NOT NULL,
    previous_adr TEXT NOT NULL,
    current_medication TEXT NOT NULL,
    systolic_bp INTEGER NOT NULL,
    diastolic_bp INTEGER NOT NULL,
    blood_sugar REAL NOT NULL,
    creatinine REAL NOT NULL,
    alt_enzyme REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER,
    drug_id INTEGER,
    drug_name TEXT NOT NULL,
    dosage_category TEXT NOT NULL DEFAULT 'Standard',
    predicted_risk TEXT NOT NULL,
    prob_low REAL NOT NULL,
    prob_moderate REAL NOT NULL,
    prob_high REAL NOT NULL,
    top_contributing_factors TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE SET NULL,
    FOREIGN KEY (drug_id) REFERENCES drugs(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS admin_users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT DEFAULT 'admin',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ocr_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_filename TEXT NOT NULL,
    extracted_text TEXT,
    matched_drug_name TEXT,
    confidence_score REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
