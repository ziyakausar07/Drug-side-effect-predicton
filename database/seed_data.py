"""
Seed Data Generator for Drug Database and Default Admin Account
Populates educational drug profiles with verified pharmacological summaries.
"""

import os
from werkzeug.security import generate_password_hash
try:
    from database.db import get_db, init_db
except ImportError:
    from db import get_db, init_db

DRUGS_DATA = [
    {
        "drug_name": "Metformin",
        "generic_name": "Metformin Hydrochloride",
        "category": "Antidiabetic (Biguanide)",
        "uses": "First-line pharmacotherapy for management of Type 2 Diabetes Mellitus and glycemic control.",
        "common_side_effects": "Diarrhea, nausea, abdominal discomfort, flatulence, metallic taste, asthenia.",
        "serious_warnings": "Lactic acidosis (rare but life-threatening); risk increases with renal impairment, sepsis, dehydration, or acute heart failure.",
        "contraindications": "Severe renal dysfunction (eGFR < 30 mL/min/1.73 m²), metabolic acidosis, acute alcohol intoxication.",
        "interactions": "Iodinated radiocontrast agents (risk of acute kidney injury), cimetidine, alcohol.",
        "precautions": "Temporarily discontinue prior to imaging with iodinated contrast media. Monitor renal function and Vitamin B12 levels annually.",
        "source_reference": "FDA Drug Label / ADA Standards of Medical Care in Diabetes"
    },
    {
        "drug_name": "Lisinopril",
        "generic_name": "Lisinopril",
        "category": "ACE Inhibitor (Antihypertensive)",
        "uses": "Treatment of essential hypertension, adjunct therapy in heart failure, and post-myocardial infarction recovery.",
        "common_side_effects": "Persistent dry non-productive cough, dizziness, headache, fatigue, postural hypotension.",
        "serious_warnings": "Angioedema (swelling of face/lips/tongue/airway), severe hyperkalemia, acute kidney injury, fetal toxicity.",
        "contraindications": "History of angioedema with ACE inhibitors, co-administration with aliskiren in diabetic patients, pregnancy (all trimesters).",
        "interactions": "Potassium-sparing diuretics, potassium supplements, NSAIDs (blunts antihypertensive effect, elevates renal risk), lithium.",
        "precautions": "Monitor serum potassium, blood pressure, and serum creatinine. Do not use in pregnancy.",
        "source_reference": "FDA Package Insert / ACC/AHA Hypertension Guidelines"
    },
    {
        "drug_name": "Amoxicillin",
        "generic_name": "Amoxicillin Trihydrate",
        "category": "Antibiotic (Aminopenicillin)",
        "uses": "Treatment of susceptible bacterial infections including respiratory tract infections, otitis media, sinusitis, and skin infections.",
        "common_side_effects": "Mild diarrhea, nausea, vomiting, erythematous maculopapular rash, headache.",
        "serious_warnings": "Severe anaphylactic reactions, Clostridioides difficile-associated diarrhea (pseudomembranous colitis), Stevens-Johnson syndrome.",
        "contraindications": "Known hypersensitivity/severe allergy to penicillins or beta-lactam antibiotics.",
        "interactions": "Methotrexate (increased toxicity), allopurinol (elevated rash incidence), oral contraceptives (marginal clearance variance), probenecid.",
        "precautions": "Inquire about prior penicillin allergy. Discontinue immediately if urticaria, wheezing, or extensive rash develops.",
        "source_reference": "USP-NF / CDC Antibiotic Prescribing Guidelines"
    },
    {
        "drug_name": "Atorvastatin",
        "generic_name": "Atorvastatin Calcium",
        "category": "Statin (HMG-CoA Reductase Inhibitor)",
        "uses": "Reduction of elevated LDL-C, total cholesterol, triglycerides, and primary/secondary prevention of cardiovascular events.",
        "common_side_effects": "Myalgia (muscle ache), arthralgia, dyspepsia, mild diarrhea, elevated transaminases.",
        "serious_warnings": "Rhabdomyolysis with acute kidney failure, immune-mediated necrotizing myopathy, severe hepatotoxicity.",
        "contraindications": "Active liver disease or unexplained persistent elevations in hepatic transaminases, pregnancy, lactation.",
        "interactions": "Strong CYP3A4 inhibitors (clarithromycin, itraconazole, ketoconazole), gemfibrozil, cyclosporine, large amounts of grapefruit juice.",
        "precautions": "Check baseline liver enzyme panels. Report unexplained muscle tenderness or dark urine promptly.",
        "source_reference": "AHA/ACC Cholesterol Clinical Guidelines / FDA Monograph"
    },
    {
        "drug_name": "Ibuprofen",
        "generic_name": "Ibuprofen",
        "category": "NSAID (Nonsteroidal Anti-inflammatory Drug)",
        "uses": "Relief of mild to moderate pain, inflammation, osteoarthritis, rheumatoid arthritis, dysmenorrhea, and fever reduction.",
        "common_side_effects": "Heartburn, nausea, epigastric pain, fluid retention, dizziness.",
        "serious_warnings": "Gastrointestinal bleeding and perforation, increased risk of serious cardiovascular thrombotic events (MI and stroke), acute renal failure.",
        "contraindications": "Active peptic ulcer disease, severe heart failure (NYHA Class IV), coronary artery bypass graft (CABG) perioperative pain, severe renal failure.",
        "interactions": "Warfarin/anticoagulants (severe hemorrhage risk), ACE inhibitors/ARBs/diuretics (reduced efficacy and nephrotoxicity), lithium, SSRIs.",
        "precautions": "Use the lowest effective dose for the shortest possible duration. Take with food or milk to minimize GI irritation.",
        "source_reference": "FDA Medication Guide / American College of Rheumatology"
    },
    {
        "drug_name": "Paracetamol",
        "generic_name": "Acetaminophen / Paracetamol",
        "category": "Analgesic / Antipyretic",
        "uses": "Temporary relief of mild to moderate pain (headache, toothache, musculoskeletal pain) and fever reduction.",
        "common_side_effects": "Rare at therapeutic doses; mild nausea, pruritus, transient elevation in ALT/AST.",
        "serious_warnings": "Severe liver injury / acute liver failure associated with acute overdose (> 4,000 mg/day in adults) or chronic alcoholism.",
        "contraindications": "Severe hepatic impairment or severe active liver disease, known hypersensitivity.",
        "interactions": "Chronic alcohol consumption (accelerates toxic metabolite NAPQI formation), warfarin (high chronic doses may increase INR), isoniazid.",
        "precautions": "Do not exceed maximum daily dosage of 4g in adults (2g in chronic liver disease/alcohol use). Check multi-ingredient cold medications.",
        "source_reference": "WHO Model List of Essential Medicines / FDA Safe Use Initiative"
    },
    {
        "drug_name": "Amlodipine",
        "generic_name": "Amlodipine Besylate",
        "category": "Calcium Channel Blocker (Dihydropyridine)",
        "uses": "Treatment of hypertension, chronic stable angina, and vasospastic (Prinzmetal's) angina.",
        "common_side_effects": "Peripheral edema (ankle swelling), flushing, palpitations, dizziness, fatigue, gingival hyperplasia.",
        "serious_warnings": "Worsening angina or acute myocardial infarction upon initiation or dosage increase (rare, severe obstructive CAD), severe hypotension.",
        "contraindications": "Severe hypotension, cardiogenic shock, clinically significant aortic stenosis.",
        "interactions": "Simvastatin (limit simvastatin dose to 20 mg), CYP3A4 inhibitors (ketoconazole, diltiazem), tacrolimus.",
        "precautions": "Caution in patients with severe hepatic impairment or congestive heart failure.",
        "source_reference": "JNC 8 Hypertension Guidelines / FDA Monograph"
    },
    {
        "drug_name": "Omeprazole",
        "generic_name": "Omeprazole",
        "category": "Proton Pump Inhibitor (PPI)",
        "uses": "Treatment of gastroesophageal reflux disease (GERD), erosive esophagitis, duodenal/gastric ulcers, Zollinger-Ellison syndrome, H. pylori eradication.",
        "common_side_effects": "Headache, abdominal pain, constipation, flatulence, diarrhea, nausea.",
        "serious_warnings": "Clostridioides difficile diarrhea, bone fractures (long-term high dose), hypomagnesemia, acute interstitial nephritis, Vitamin B12 deficiency.",
        "contraindications": "Known hypersensitivity to substituted benzimidazoles; co-administration with rilpivirine.",
        "interactions": "Clopidogrel (decreased antiplatelet efficacy via CYP2C19 inhibition), diazepam, phenytoin, methotrexate, antiretrovirals.",
        "precautions": "Review long-term use indications periodically. Maintain adequate calcium and vitamin D intake.",
        "source_reference": "American Gastroenterological Association Clinical Guidelines"
    },
    {
        "drug_name": "Ciprofloxacin",
        "generic_name": "Ciprofloxacin Hydrochloride",
        "category": "Fluoroquinolone Antibiotic",
        "uses": "Treatment of complicated urinary tract infections, severe respiratory infections, intra-abdominal infections, and bacterial prostatitis.",
        "common_side_effects": "Nausea, vomiting, diarrhea, dizziness, headache, sleep disturbance.",
        "serious_warnings": "Tendinitis and tendon rupture (especially Achilles), peripheral neuropathy, CNS effects (hallucinations, tremors, seizures), QT prolongation, aortic aneurysm rupture.",
        "contraindications": "Hypersensitivity to quinolones, co-administration with tizanidine.",
        "interactions": "Theophylline (elevated levels), oral antacids/sucralfate/multivalent cations (drastically reduced absorption), warfarin, QT-prolonging drugs.",
        "precautions": "Reserve for infections where alternative antibacterials are inappropriate. Avoid strenuous exercise during treatment.",
        "source_reference": "FDA Black Box Warnings / Infectious Diseases Society of America (IDSA)"
    },
    {
        "drug_name": "Sertraline",
        "generic_name": "Sertraline Hydrochloride",
        "category": "SSRI (Selective Serotonin Reuptake Inhibitor)",
        "uses": "Treatment of Major Depressive Disorder (MDD), Obsessive-Compulsive Disorder (OCD), Panic Disorder, PTSD, Social Anxiety Disorder.",
        "common_side_effects": "Nausea, insomnia, diarrhea, sexual dysfunction, somnolence, tremor, dry mouth, sweating.",
        "serious_warnings": "Suicidal ideation in children/young adults (black box warning), Serotonin Syndrome, hyponatremia/SIADH, increased bleeding risk.",
        "contraindications": "Concomitant use of MAOIs (risk of fatal serotonin syndrome), pimozide, disulfiram (liquid concentrate containing alcohol).",
        "interactions": "MAOIs, linezolid, methylene blue, NSAIDs/aspirin (elevated bleeding), triptans, tramadol, St. John's wort.",
        "precautions": "Do not discontinue abruptly (taper gradually to avoid discontinuation syndrome). Monitor mood changes closely.",
        "source_reference": "APA Practice Guidelines for Major Depressive Disorder"
    },
    {
        "drug_name": "Metoprolol",
        "generic_name": "Metoprolol Tartrate / Succinate",
        "category": "Beta-1 Selective Adrenergic Receptor Blocker",
        "uses": "Management of hypertension, angina pectoris, hemodynamically stable acute myocardial infarction, and chronic heart failure.",
        "common_side_effects": "Bradycardia, fatigue, dizziness, cold extremities, exercise intolerance, hypotension.",
        "serious_warnings": "Severe bradycardia, heart block, cardiogenic shock, bronchospasm in asthmatic patients, masking of hypoglycemia symptoms.",
        "contraindications": "Sinus bradycardia (< 45 bpm), 2nd or 3rd-degree AV block, decompensated cardiogenic shock, sick sinus syndrome.",
        "interactions": "CYP2D6 inhibitors (fluoxetine, paroxetine), diltiazem, verapamil, digoxin, clonidine (rebound hypertension upon withdrawal).",
        "precautions": "Never abruptly withdraw therapy in ischemic heart disease. Exercise caution in reactive airway disease.",
        "source_reference": "ACC/AHA Guideline for the Management of Heart Failure"
    },
    {
        "drug_name": "Aspirin",
        "generic_name": "Acetylsalicylic Acid (ASA)",
        "category": "Antiplatelet / Salicylate",
        "uses": "Secondary prevention of cardiovascular events (MI, ischemic stroke), acute coronary syndrome, relief of mild-to-moderate pain and inflammation.",
        "common_side_effects": "Dyspepsia, heartburn, epigastric discomfort, easy bruising, minor bleeding.",
        "serious_warnings": "Major gastrointestinal hemorrhage, intracranial bleeding, Reye's syndrome in children/teens with viral illness, severe salicylate toxicity.",
        "contraindications": "Aspirin-exacerbated respiratory disease (AERD), bleeding disorders (hemophilia), active peptic ulcer, children/adolescents with viral infection.",
        "interactions": "Anticoagulants (warfarin, DOACs), other NSAIDs (ibuprofen blocks aspirin antiplatelet effect if taken prior), SSRIs, methotrexate.",
        "precautions": "Discontinue 5-7 days prior to elective surgery if requested by surgical team. Take with food.",
        "source_reference": "USPSTF Prevention Guidelines / ACC/AHA Prevention Guidelines"
    },
    {
        "drug_name": "Warfarin",
        "generic_name": "Warfarin Sodium",
        "category": "Anticoagulant (Vitamin K Antagonist)",
        "uses": "Prophylaxis and treatment of venous thromboembolism, pulmonary embolism, thromboembolic complications associated with atrial fibrillation or mechanical heart valves.",
        "common_side_effects": "Minor bleeding (epistaxis, gingival bleeding), bruising, alopecia, skin necrosis (rare).",
        "serious_warnings": "Major fatal hemorrhage, intracranial hemorrhage, calciphylaxis, high risk of teratogenicity in pregnancy.",
        "contraindications": "Pregnancy (except mechanical valves with high risk), active major bleeding, severe thrombocytopenia, uncontrolled severe hypertension.",
        "interactions": "Extensive interactions: NSAIDs, antibiotics (ciprofloxacin, metronidazole), amiodarone, leafy greens/vitamin K foods, St. John's wort.",
        "precautions": "Regular monitoring of INR (target usually 2.0-3.0) is mandatory. Maintain consistent dietary intake of Vitamin K.",
        "source_reference": "CHEST Guidelines for Antithrombotic Therapy / FDA Monograph"
    },
    {
        "drug_name": "Prednisone",
        "generic_name": "Prednisone",
        "category": "Systemic Corticosteroid (Glucocorticoid)",
        "uses": "Management of acute inflammatory, allergic, and autoimmune conditions (rheumatoid arthritis, asthma exacerbation, inflammatory bowel disease).",
        "common_side_effects": "Insomnia, mood changes, increased appetite, weight gain, fluid retention, mild hypertension.",
        "serious_warnings": "Hyperglycemia / steroid-induced diabetes, adrenal suppression, osteoporosis/osteonecrosis, immunosuppression/opportunistic infections, peptic ulceration.",
        "contraindications": "Systemic fungal infections, known hypersensitivity, live virus vaccines during immunosuppressive doses.",
        "interactions": "NSAIDs (exponential increase in ulcer/GI bleed risk), antidiabetic agents (antagonizes glycemic control), phenytoin, rifampin.",
        "precautions": "Do not stop abruptly after prolonged use (> 2 weeks); taper gradually to permit HPA axis recovery. Monitor blood sugar and blood pressure.",
        "source_reference": "ACR Guidelines / Endocrine Society Clinical Guidelines"
    },
    {
        "drug_name": "Azithromycin",
        "generic_name": "Azithromycin Dihydrate",
        "category": "Macrolide Antibiotic (Azalide)",
        "uses": "Treatment of mild to moderate community-acquired pneumonia, acute bacterial exacerbations of COPD, pharyngitis, skin infections, chlamydia.",
        "common_side_effects": "Diarrhea, nausea, abdominal cramping, vomiting, headache.",
        "serious_warnings": "QT interval prolongation and torsades de pointes (ventricular arrhythmia), cholestatic jaundice/hepatic dysfunction, Clostridioides difficile colitis.",
        "contraindications": "History of cholestatic jaundice/hepatic dysfunction associated with prior azithromycin use, macrolide hypersensitivity.",
        "interactions": "QT-prolonging agents (antiarrhythmics, antipsychotics), digoxin (elevated digoxin levels), antacids containing aluminum/magnesium.",
        "precautions": "Avoid in patients with known prolongation of QT interval or uncorrected hypokalemia/hypomagnesemia.",
        "source_reference": "IDSA Guidelines / FDA Drug Safety Communication"
    }
]


def seed_database():
    init_db()
    conn = get_db()
    cursor = conn.cursor()
    
    # 1. Seed Drugs
    print("[*] Seeding educational drug catalog...")
    for drug in DRUGS_DATA:
        cursor.execute("""
            INSERT OR REPLACE INTO drugs 
            (drug_name, generic_name, category, uses, common_side_effects, serious_warnings, contraindications, interactions, precautions, source_reference)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            drug["drug_name"],
            drug["generic_name"],
            drug["category"],
            drug["uses"],
            drug["common_side_effects"],
            drug["serious_warnings"],
            drug["contraindications"],
            drug["interactions"],
            drug["precautions"],
            drug["source_reference"]
        ))
        
    # 2. Seed Default Admin
    print("[*] Creating default admin account...")
    admin_user = "admin"
    admin_pass = "admin123"
    pass_hash = generate_password_hash(admin_pass)
    
    cursor.execute("""
        INSERT OR IGNORE INTO admin_users (username, password_hash, role)
        VALUES (?, ?, 'admin')
    """, (admin_user, pass_hash))
    
    # 3. Seed Sample Patient and Prediction Logs for Instant Demo
    cursor.execute("SELECT COUNT(*) as cnt FROM predictions")
    if cursor.fetchone()['cnt'] == 0:
        print("[*] Seeding initial sample prediction logs for dashboard demo...")
        sample_patients = [
            (68, 'Male', 82.5, 'Hypertension', 'ACE Inhibitors', 'Yes', 'Diuretics', 155, 95, 110.0, 1.6, 32.0),
            (45, 'Female', 65.0, 'Type 2 Diabetes', 'None', 'No', 'Metformin', 122, 78, 160.0, 0.9, 24.0),
            (72, 'Male', 76.0, 'Chronic Kidney Disease', 'Penicillin', 'Yes', 'Beta-Blockers', 140, 85, 125.0, 2.3, 40.0),
            (29, 'Female', 58.0, 'None', 'None', 'No', 'None', 115, 75, 92.0, 0.8, 18.0),
            (62, 'Female', 70.0, 'Asthma', 'NSAIDs / Aspirin', 'Yes', 'PPIs', 130, 80, 105.0, 1.1, 28.0)
        ]
        
        sample_preds = [
            (1, 2, 'Lisinopril', 'Standard', 'High', 0.08, 0.22, 0.70, 'Known ACE Inhibitor Allergy, Elevated Creatinine (1.6 mg/dL), Age >= 65'),
            (2, 1, 'Metformin', 'Standard', 'Low', 0.85, 0.12, 0.03, 'Normal Renal Function, No Known Allergies'),
            (3, 9, 'Ciprofloxacin', 'Standard', 'High', 0.10, 0.25, 0.65, 'Chronic Kidney Disease, Elevated Creatinine (2.3 mg/dL), Prior ADR History'),
            (4, 6, 'Paracetamol', 'Standard', 'Low', 0.92, 0.06, 0.02, 'Young Adult, Normal Hepatic Function, Low Baseline Risk'),
            (5, 5, 'Ibuprofen', 'Standard', 'High', 0.05, 0.15, 0.80, 'Known NSAID Allergy, Prior Adverse Reaction')
        ]
        
        for pat in sample_patients:
            cursor.execute("""
                INSERT INTO patients 
                (age, gender, weight, medical_condition, allergy_history, previous_adr, current_medication, systolic_bp, diastolic_bp, blood_sugar, creatinine, alt_enzyme)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, pat)
            
        for pred in sample_preds:
            cursor.execute("""
                INSERT INTO predictions
                (patient_id, drug_id, drug_name, dosage_category, predicted_risk, prob_low, prob_moderate, prob_high, top_contributing_factors)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, pred)
            
    conn.commit()
    print("[+] Database seeding completed successfully.")


if __name__ == "__main__":
    seed_database()
