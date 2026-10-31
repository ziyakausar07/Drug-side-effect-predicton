"""
Import dataset from E:\Project\Drug Side Effects\Drug_side_effect
Populates SQLite database with comprehensive drugs catalog and medical conditions.
"""

import os
import sys
import re
import json
import sqlite3
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'database', 'database.db')
SOURCE_DIR = r"E:\Project\Drug Side Effects\Drug_side_effect"
CSV_DRUGS_COM = os.path.join(SOURCE_DIR, "drugs_side_effects_drugs_com.csv")
CSV_MED_DETAILS = os.path.join(SOURCE_DIR, "Medicine_Details.csv")

def clean_text(val, fallback=""):
    if pd.isna(val) or val is None:
        return fallback
    text = str(val).strip()
    return text if text else fallback

def run_import():
    print("[*] Connecting to database:", DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Create tables if not exist
    cur.execute("""
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
            source_reference TEXT DEFAULT 'FDA / DailyMed / Drugs.com',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS medical_conditions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            condition_name TEXT NOT NULL UNIQUE,
            category TEXT DEFAULT 'General',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    # Track seen drugs and conditions
    seen_drugs = set()
    conditions_set = set()

    # Query already existing drugs
    cur.execute("SELECT LOWER(drug_name) FROM drugs")
    for row in cur.fetchall():
        seen_drugs.add(row[0].strip().lower())

    drugs_to_insert = []

    # 1. Import from drugs_side_effects_drugs_com.csv
    if os.path.exists(CSV_DRUGS_COM):
        print(f"[*] Reading {CSV_DRUGS_COM}...")
        df1 = pd.read_csv(CSV_DRUGS_COM)
        for _, row in df1.iterrows():
            raw_name = clean_text(row.get('drug_name'))
            if not raw_name:
                continue
            
            # Format nicely
            drug_name = raw_name.strip()
            # If all lowercase, title-case it nicely
            if drug_name.islower():
                drug_name = drug_name.title()

            key = drug_name.lower()
            if key in seen_drugs:
                continue
            seen_drugs.add(key)

            generic_name = clean_text(row.get('generic_name'), drug_name)
            if generic_name.islower():
                generic_name = generic_name.title()

            category = clean_text(row.get('drug_classes'), 'Therapeutic Agent')
            # Extract first primary class if long
            if ';' in category:
                category = category.split(';')[0].strip()

            condition = clean_text(row.get('medical_condition'))
            if condition:
                conditions_set.add(condition.strip())

            cond_desc = clean_text(row.get('medical_condition_description'))
            uses_str = f"Indicated for {condition}. {cond_desc}".strip() if condition else cond_desc
            if not uses_str:
                uses_str = f"Pharmacological therapy with {generic_name}."

            side_effects = clean_text(row.get('side_effects'), "Mild gastrointestinal discomfort, headache, or fatigue.")
            
            preg = clean_text(row.get('pregnancy_category'), 'Unclassified')
            csa = clean_text(row.get('csa'), 'Not a controlled substance')
            warnings = f"Pregnancy Category: {preg}. Controlled Substance: {csa}. Consult physician if unexpected severe reactions occur."
            
            contraindications = f"Known hypersensitivity to {generic_name} or related pharmaceutical formulations."
            
            alc = clean_text(row.get('alcohol'), 'Consult physician')
            related = clean_text(row.get('related_drugs'), 'None noted')
            if len(related) > 200:
                related = related[:200] + "..."
            interactions = f"Alcohol: {alc}; Related compounds: {related}"
            
            rx = clean_text(row.get('rx_otc'), 'Prescription / OTC')
            precautions = f"Status: {rx}. Check renal and hepatic function during extended therapy."

            drugs_to_insert.append((
                drug_name, generic_name, category, uses_str, side_effects,
                warnings, contraindications, interactions, precautions,
                'Drugs.com Comprehensive Database / FDA'
            ))
        print(f"[+] Prepared {len(drugs_to_insert)} drugs from Drugs.com dataset.")

    # 2. Import from Medicine_Details.csv (add branded medications)
    if os.path.exists(CSV_MED_DETAILS):
        print(f"[*] Reading {CSV_MED_DETAILS}...")
        df2 = pd.read_csv(CSV_MED_DETAILS)
        med_count = 0
        for _, row in df2.iterrows():
            raw_name = clean_text(row.get('Medicine Name'))
            if not raw_name:
                continue

            drug_name = raw_name.strip()
            key = drug_name.lower()
            if key in seen_drugs:
                continue
            seen_drugs.add(key)

            composition = clean_text(row.get('Composition'), drug_name)
            uses = clean_text(row.get('Uses'), 'General therapeutic treatment')
            
            # Extract conditions from Uses
            for part in uses.split(';'):
                part = re.sub(r'^(Treatment of|Prevention of)\s+', '', part.strip(), flags=re.I).strip()
                if 2 < len(part) < 60 and '</span>' not in part:
                    conditions_set.add(part)

            side_effects = clean_text(row.get('Side_effects'), 'Mild headache, nausea, abdominal discomfort')
            manufacturer = clean_text(row.get('Manufacturer'), 'Licensed Pharmaceutical Manufacturer')

            category = "Pharmaceutical Formulation"
            warnings = "Discontinue use and seek urgent medical attention if allergic anaphylaxis occurs."
            contraindications = f"Known severe hypersensitivity to {composition}."
            interactions = "Inform attending healthcare physician of all concurrent medications."
            precautions = f"Manufactured by {manufacturer}. Store in cool, dry place away from direct sunlight."

            drugs_to_insert.append((
                drug_name, composition, category, uses, side_effects,
                warnings, contraindications, interactions, precautions,
                f'National Formulary / {manufacturer}'
            ))
            med_count += 1

        print(f"[+] Added {med_count} additional medicines from Medicine_Details.")

    # Insert into database in batches
    print(f"[*] Inserting total {len(drugs_to_insert)} new drugs into SQLite...")
    cur.executemany("""
        INSERT OR IGNORE INTO drugs (drug_name, generic_name, category, uses, common_side_effects, serious_warnings, contraindications, interactions, precautions, source_reference)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, drugs_to_insert)
    conn.commit()

    # 3. Medical conditions compilation
    # Add established clinical conditions
    standard_conditions = [
        "None",
        "Acne",
        "ADHD (Attention Deficit Hyperactivity Disorder)",
        "AIDS / HIV Infection",
        "Allergies / Allergic Rhinitis",
        "Alzheimer's Disease",
        "Angina Pectoris",
        "Anxiety Disorder",
        "Arrhythmia / Cardiac Dysrhythmia",
        "Asthma",
        "Bacterial Infections",
        "Bipolar Disorder",
        "Bronchitis (Acute & Chronic)",
        "Cancer / Oncology Therapy",
        "Chronic Kidney Disease (CKD)",
        "Chronic Obstructive Pulmonary Disease (COPD)",
        "Chronic Pain Syndrome",
        "Colds & Flu (Viral Respiratory)",
        "Constipation",
        "Covid-19 Infection",
        "Depression (Major Depressive Disorder)",
        "Diabetes Mellitus (Type 1)",
        "Diabetes Mellitus (Type 2)",
        "Diarrhea / Gastroenteritis",
        "Dyslipidemia / High Cholesterol",
        "Eczema / Atopic Dermatitis",
        "Emergency Contraception",
        "Epilepsy / Seizure Disorder",
        "Erectile Dysfunction",
        "Fibromyalgia",
        "Fungal Infections",
        "Gastroesophageal Reflux Disease (GERD)",
        "Glaucoma / Ocular Hypertension",
        "Gout / Hyperuricemia",
        "Heart Failure (Congestive Heart Failure)",
        "Herpes Simplex / Herpes Zoster",
        "Hypertension (High Blood Pressure)",
        "Hyperthyroidism",
        "Hypothyroidism",
        "Inflammatory Bowel Disease (Crohn's / Colitis)",
        "Insomnia / Sleep Disorders",
        "Migraine Headache",
        "Multiple Sclerosis",
        "Muscle Spasms / Musculoskeletal Pain",
        "Nausea & Vomiting",
        "Neuropathic Pain",
        "Obesity / Weight Management",
        "Osteoarthritis",
        "Osteoporosis",
        "Panic Disorder",
        "Parkinson's Disease",
        "Peptic Ulcer Disease",
        "Pneumonia",
        "Psoriasis",
        "Rheumatoid Arthritis",
        "Schizophrenia",
        "Sinusitis",
        "Stroke / Cerebrovascular Disease",
        "Tuberculosis",
        "Urinary Tract Infection (UTI)"
    ]

    all_conditions = set(standard_conditions)
    # Add extracted clean conditions
    for c in conditions_set:
        c_clean = c.strip()
        if 2 < len(c_clean) < 60 and not re.search(r'[0-9]{2,}', c_clean):
            # Title case if all lower
            if c_clean.islower():
                c_clean = c_clean.title()
            all_conditions.add(c_clean)

    # Clean and sort
    sorted_conditions = sorted(list(all_conditions), key=lambda x: (x != "None", x.lower()))

    # Insert medical conditions into database
    cur.execute("DELETE FROM medical_conditions")
    for cond in sorted_conditions:
        cur.execute("INSERT OR IGNORE INTO medical_conditions (condition_name) VALUES (?)", (cond,))
    conn.commit()

    # Save to a conditions json file for rapid template access
    conditions_json_path = os.path.join(BASE_DIR, 'database', 'conditions.json')
    with open(conditions_json_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_conditions, f, indent=2)

    # Verification counts
    cur.execute("SELECT COUNT(*) FROM drugs")
    total_drugs = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM medical_conditions")
    total_conditions = cur.fetchone()[0]

    conn.close()

    print("=" * 60)
    print(f"[✓] Import Complete!")
    print(f"    Total Drugs in Catalog: {total_drugs}")
    print(f"    Total Medical Conditions: {total_conditions}")
    print(f"    Conditions cached at: {conditions_json_path}")
    print("=" * 60)

if __name__ == "__main__":
    run_import()
