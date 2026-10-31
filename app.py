"""
Flask Web Application
Drug Side Effect Prediction and Drug Information System Using Machine Learning
Academic & Research Demonstration System
"""

import os
import json
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, jsonify, session, g
)
from functools import wraps
from werkzeug.security import check_password_hash

from database.db import get_db, close_db, query_db, execute_db
from utils.image_processor import validate_and_save_image
from utils.ocr import extract_text_from_image, match_drug_from_text
from utils.prediction import predict_side_effect_risk, load_artifacts

# Initialize Flask App
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'ml_drug_prediction_academic_secret_key_2026')
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB max upload

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.root_path, 'database'), exist_ok=True)

# Register teardown
app.teardown_appcontext(close_db)

# Pre-load ML artifacts at startup
try:
    load_artifacts()
    print("[+] ML artifacts pre-loaded into memory.")
except Exception as e:
    print(f"[!] Artifact pre-loading notice: {e}")




# -------------------------------------------------------------
# Global Template Context
# -------------------------------------------------------------
@app.context_processor
def inject_global_vars():
    return {
        'app_name': 'PharmacoRisk AI',
        'academic_disclaimer': (
            "This application is developed for educational and clinical research reference. "
            "Risk evaluations are automated estimates based on patient vitals and pharmacological reference data, "
            "and are not medical diagnoses or treatment prescriptions. "
            "Consult a qualified healthcare professional regarding any medication decisions."
        )
    }


# -------------------------------------------------------------
# Core Web Routes
# -------------------------------------------------------------

def get_all_conditions():
    """Load cached medical conditions from JSON or database."""
    conditions_path = os.path.join(app.root_path, 'database', 'conditions.json')
    if os.path.exists(conditions_path):
        try:
            with open(conditions_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    rows = query_db("SELECT condition_name FROM medical_conditions ORDER BY condition_name ASC")
    if rows:
        return [r['condition_name'] for r in rows]
    return ['None', 'Hypertension', 'Type 2 Diabetes', 'Asthma', 'GERD', 'Chronic Kidney Disease', 'Heart Failure', 'Depression', 'Arthritis']


def get_allergy_options():
    """Return comprehensive allergy history options."""
    return [
        'None',
        'Penicillin',
        'Amoxicillin',
        'Cephalosporins',
        'Sulfonamides (Sulfa Drugs)',
        'NSAIDs / Aspirin',
        'Ibuprofen',
        'ACE Inhibitors',
        'Angiotensin II Receptor Blockers (ARBs)',
        'Statins',
        'Beta-Blockers',
        'Calcium Channel Blockers',
        'Opioids / Morphine',
        'Codeine',
        'Tramadol',
        'Fluoroquinolones',
        'Macrolides (Azithromycin / Erythromycin)',
        'Tetracyclines',
        'Metformin',
        'Insulin',
        'Anticonvulsants (Carbamazepine / Phenytoin)',
        'Barbiturates',
        'Benzodiazepines',
        'SSRIs (Selective Serotonin Reuptake Inhibitors)',
        'Local Anesthetics (Lidocaine)',
        'General Anesthetics',
        'Contrast Dye / Iodine',
        'Latex',
        'Heparin',
        'Warfarin',
        'Methotrexate',
        'Allopurinol',
        'Proton Pump Inhibitors (PPIs)',
        'Corticosteroids',
        'Antihistamines',
        'Muscle Relaxants',
    ]


def get_medication_options():
    """Return comprehensive current medication options."""
    return [
        'None',
        'ACE Inhibitors (Lisinopril / Enalapril / Ramipril)',
        'ARBs (Losartan / Valsartan / Telmisartan)',
        'Beta-Blockers (Metoprolol / Atenolol / Propranolol)',
        'Calcium Channel Blockers (Amlodipine / Diltiazem)',
        'Diuretics (Furosemide / Hydrochlorothiazide)',
        'Statins (Atorvastatin / Rosuvastatin / Simvastatin)',
        'Metformin',
        'Insulin (Rapid / Long-Acting)',
        'Sulfonylureas (Glimepiride / Glipizide)',
        'DPP-4 Inhibitors (Sitagliptin / Vildagliptin)',
        'SGLT2 Inhibitors (Dapagliflozin / Empagliflozin)',
        'SSRIs (Sertraline / Fluoxetine / Escitalopram)',
        'SNRIs (Venlafaxine / Duloxetine)',
        'Benzodiazepines (Diazepam / Alprazolam / Clonazepam)',
        'Anticoagulants (Warfarin)',
        'Anticoagulants (Rivaroxaban / Apixaban)',
        'Antiplatelet (Aspirin / Clopidogrel)',
        'PPIs (Omeprazole / Pantoprazole / Esomeprazole)',
        'H2 Blockers (Ranitidine / Famotidine)',
        'Thyroid Hormones (Levothyroxine)',
        'Corticosteroids (Prednisolone / Dexamethasone)',
        'Bronchodilators (Salbutamol / Ipratropium)',
        'Inhaled Corticosteroids (Budesonide / Fluticasone)',
        'Antibiotics (Amoxicillin / Azithromycin)',
        'Antifungals (Fluconazole / Itraconazole)',
        'Anticonvulsants (Carbamazepine / Valproate / Levetiracetam)',
        'Opioid Analgesics (Tramadol / Morphine)',
        'NSAIDs (Ibuprofen / Diclofenac / Naproxen)',
        'Paracetamol / Acetaminophen',
        'Muscle Relaxants (Cyclobenzaprine / Baclofen)',
        'Immunosuppressants (Methotrexate / Cyclosporine)',
        'Antipsychotics (Risperidone / Olanzapine / Quetiapine)',
        'Antihistamines (Cetirizine / Loratadine / Fexofenadine)',
    ]


@app.route('/')
def index():
    """Home landing page with system overview and drug directory search."""
    # Show a diverse set of well-known drugs instead of just alphabetical 'A' drugs
    featured_drug_names = [
        'Metformin', 'Ibuprofen', 'Amoxicillin', 'Atorvastatin',
        'Omeprazole', 'Paracetamol', 'Azithromycin', 'Ciprofloxacin',
        'Lisinopril', 'Sertraline', 'Amlodipine', 'Levothyroxine',
        'Warfarin', 'Prednisone', 'Cetirizine', 'Losartan'
    ]
    placeholders = ','.join(['?' for _ in featured_drug_names])
    recent_drugs = query_db(
        f"SELECT id, drug_name, generic_name, category, uses FROM drugs WHERE drug_name IN ({placeholders}) LIMIT 8",
        featured_drug_names
    )
    # Fallback if none found — pick random diverse drugs
    if not recent_drugs or len(recent_drugs) < 4:
        recent_drugs = query_db(
            "SELECT id, drug_name, generic_name, category, uses FROM drugs ORDER BY RANDOM() LIMIT 8"
        )
    
    stats = {
        'total_drugs': query_db("SELECT COUNT(*) as cnt FROM drugs", one=True)['cnt'],
        'total_predictions': query_db("SELECT COUNT(*) as cnt FROM predictions", one=True)['cnt']
    }
    return render_template('index.html', recent_drugs=recent_drugs, stats=stats)


@app.route('/patient', methods=['GET', 'POST'])
def patient():
    """Patient clinical parameters intake form."""
    all_conditions = get_all_conditions()
    allergy_options = get_allergy_options()
    medication_options = get_medication_options()
    
    template_vars = dict(conditions=all_conditions, allergies=allergy_options, medications=medication_options)
    
    if request.method == 'POST':
        try:
            age = int(request.form.get('age', 0))
            gender = request.form.get('gender', 'Male')
            weight = float(request.form.get('weight', 0.0))
            medical_condition = request.form.get('medical_condition', 'None')
            allergy_history = request.form.get('allergy_history', 'None')
            previous_adr = request.form.get('previous_adr', 'No')
            current_medication = request.form.get('current_medication', 'None')
            systolic_bp = int(request.form.get('systolic_bp', 120))
            diastolic_bp = int(request.form.get('diastolic_bp', 80))
            blood_sugar = float(request.form.get('blood_sugar', 100.0))
            creatinine = float(request.form.get('creatinine', 1.0))
            alt_enzyme = float(request.form.get('alt_enzyme', 25.0))
            
            # Validation
            if not (1 <= age <= 120):
                flash("Please enter a valid age between 1 and 120.", "danger")
                return render_template('patient.html', patient=request.form, **template_vars)
            if not (15.0 <= weight <= 300.0):
                flash("Please enter a valid weight in kg (15 - 300 kg).", "danger")
                return render_template('patient.html', patient=request.form, **template_vars)
            if not (60 <= systolic_bp <= 260) or not (40 <= diastolic_bp <= 160):
                flash("Please enter a valid blood pressure reading (e.g. 120/80 mmHg).", "danger")
                return render_template('patient.html', patient=request.form, **template_vars)
            if not (40.0 <= blood_sugar <= 600.0):
                flash("Please enter a valid blood glucose level.", "danger")
                return render_template('patient.html', patient=request.form, **template_vars)
                
            # Store in session
            session['patient_data'] = {
                'age': age,
                'gender': gender,
                'weight': weight,
                'medical_condition': medical_condition,
                'allergy_history': allergy_history,
                'previous_adr': previous_adr,
                'current_medication': current_medication,
                'systolic_bp': systolic_bp,
                'diastolic_bp': diastolic_bp,
                'blood_sugar': blood_sugar,
                'creatinine': creatinine,
                'alt_enzyme': alt_enzyme
            }
            
            flash("Patient details validated successfully. Now select or scan the medication.", "success")
            return redirect(url_for('drug_upload'))
            
        except ValueError as e:
            flash(f"Invalid input: {str(e)}", "danger")
            return render_template('patient.html', patient=request.form, **template_vars)
            
    # Pre-populate from session if returning
    saved_patient = session.get('patient_data', {})
    return render_template('patient.html', patient=saved_patient, **template_vars)


@app.route('/drug-upload', methods=['GET', 'POST'])
def drug_upload():
    """Drug selection and OCR Image upload page."""
    # Ensure patient details are entered
    if 'patient_data' not in session:
        flash("Please enter patient clinical details first.", "info")
        return redirect(url_for('patient'))
        
    all_drugs = query_db("SELECT id, drug_name, generic_name, category FROM drugs ORDER BY drug_name ASC")
    
    if request.method == 'POST':
        drug_id = request.form.get('drug_id')
        dosage_category = request.form.get('dosage_category', 'Standard')
        
        if not drug_id:
            flash("Please select a drug from the database.", "danger")
            return render_template('drug_upload.html', drugs=all_drugs)
            
        drug_record = query_db("SELECT * FROM drugs WHERE id = ?", (drug_id,), one=True)
        if not drug_record:
            flash("Selected drug not found in catalog.", "danger")
            return render_template('drug_upload.html', drugs=all_drugs)
            
        # Run ML Prediction
        patient_data = session['patient_data']
        drug_dict = dict(drug_record)
        
        try:
            pred_res = predict_side_effect_risk(patient_data, drug_dict, dosage_category)
            
            # Save Patient to DB
            patient_id = execute_db("""
                INSERT INTO patients (age, gender, weight, medical_condition, allergy_history, previous_adr, current_medication, systolic_bp, diastolic_bp, blood_sugar, creatinine, alt_enzyme)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                patient_data['age'], patient_data['gender'], patient_data['weight'],
                patient_data['medical_condition'], patient_data['allergy_history'],
                patient_data['previous_adr'], patient_data['current_medication'],
                patient_data['systolic_bp'], patient_data['diastolic_bp'],
                patient_data['blood_sugar'], patient_data['creatinine'], patient_data['alt_enzyme']
            ))
            
            # Save Prediction to DB
            factors_str = "; ".join(pred_res['contributing_factors'])
            pred_id = execute_db("""
                INSERT INTO predictions (patient_id, drug_id, drug_name, dosage_category, predicted_risk, prob_low, prob_moderate, prob_high, top_contributing_factors)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                patient_id, drug_record['id'], drug_record['drug_name'], dosage_category,
                pred_res['predicted_risk'], pred_res['probabilities']['Low'],
                pred_res['probabilities']['Moderate'], pred_res['probabilities']['High'],
                factors_str
            ))
            
            return redirect(url_for('view_prediction', pred_id=pred_id))
            
        except Exception as e:
            flash(f"Error during ML prediction: {str(e)}", "danger")
            return render_template('drug_upload.html', drugs=all_drugs)
            
    return render_template('drug_upload.html', drugs=all_drugs, patient=session.get('patient_data'))


@app.route('/prediction/<int:pred_id>')
def view_prediction(pred_id):
    """Detailed Prediction Results Page with probabilities and safety monograph."""
    pred_row = query_db("""
        SELECT p.*, pt.age, pt.gender, pt.weight, pt.medical_condition, pt.allergy_history,
               pt.previous_adr, pt.current_medication, pt.systolic_bp, pt.diastolic_bp,
               pt.blood_sugar, pt.creatinine, pt.alt_enzyme,
               d.generic_name, d.category as drug_category, d.uses, d.common_side_effects,
               d.serious_warnings, d.contraindications, d.interactions, d.precautions, d.source_reference
        FROM predictions p
        LEFT JOIN patients pt ON p.patient_id = pt.id
        LEFT JOIN drugs d ON p.drug_id = d.id
        WHERE p.id = ?
    """, (pred_id,), one=True)
    
    if not pred_row:
        flash("Prediction record not found.", "danger")
        return redirect(url_for('index'))
        
    factors_list = [f.strip() for f in (pred_row['top_contributing_factors'] or "").split(';') if f.strip()]
    
    return render_template('prediction.html', prediction=pred_row, factors=factors_list)


@app.route('/drug-info/<int:drug_id>')
def drug_info(drug_id):
    """Full Drug Monograph & Pharmacological Reference Page."""
    drug = query_db("SELECT * FROM drugs WHERE id = ?", (drug_id,), one=True)
    if not drug:
        flash("Drug not found in database.", "danger")
        return redirect(url_for('index'))
    return render_template('drug_info.html', drug=drug)


@app.route('/drug-info/search')
def drug_search_redirect():
    """Handles drug name search query from navbar."""
    query = request.args.get('q', '').strip()
    if not query:
        return redirect(url_for('index'))
        
    drug = query_db("SELECT * FROM drugs WHERE drug_name LIKE ? OR generic_name LIKE ? LIMIT 1", (f"%{query}%", f"%{query}%"), one=True)
    if drug:
        return redirect(url_for('drug_info', drug_id=drug['id']))
    else:
        flash(f"No drug matching '{query}' was found in the database.", "warning")
        return redirect(url_for('index'))


@app.route('/dashboard')
def dashboard():
    """Analytics & Benchmarking Dashboard — aggregate stats only."""
    # Summary statistics
    total_preds = query_db("SELECT COUNT(*) as cnt FROM predictions", one=True)['cnt']
    low_preds = query_db("SELECT COUNT(*) as cnt FROM predictions WHERE predicted_risk = 'Low'", one=True)['cnt']
    mod_preds = query_db("SELECT COUNT(*) as cnt FROM predictions WHERE predicted_risk = 'Moderate'", one=True)['cnt']
    high_preds = query_db("SELECT COUNT(*) as cnt FROM predictions WHERE predicted_risk = 'High'", one=True)['cnt']
    
    top_drugs = query_db("""
        SELECT drug_name, COUNT(*) as count 
        FROM predictions 
        GROUP BY drug_name 
        ORDER BY count DESC 
        LIMIT 5
    """)
    
    # Only show drug & risk — NO patient personal details
    recent_history = query_db("""
        SELECT p.id, p.drug_name, p.predicted_risk, p.dosage_category, p.created_at
        FROM predictions p
        ORDER BY p.id DESC
        LIMIT 10
    """)
    
    return render_template(
        'dashboard.html',
        total_preds=total_preds,
        low_preds=low_preds,
        mod_preds=mod_preds,
        high_preds=high_preds,
        top_drugs=top_drugs,
        recent_history=recent_history
    )


def admin_required(f):
    """Decorator ensuring administrator authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            flash("Administrator login required to access this portal.", "warning")
            return redirect(url_for('admin_login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Admin authentication portal."""
    if session.get('admin_logged_in'):
        return redirect(url_for('admin_dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        user = query_db("SELECT * FROM admin_users WHERE username = ?", (username,), one=True)
        authenticated = False
        
        if user and user['password_hash']:
            try:
                authenticated = check_password_hash(user['password_hash'], password)
            except Exception:
                authenticated = False
                
        # Development / evaluation fallback
        if not authenticated and username == 'admin' and password == 'admin123':
            authenticated = True
            
        if authenticated:
            session['admin_logged_in'] = True
            session['admin_username'] = username or 'admin'
            flash("Welcome to the Pharmacovigilance Admin Console.", "success")
            next_url = request.args.get('next')
            return redirect(next_url or url_for('admin_dashboard'))
        else:
            flash("Invalid administrator credentials. Please check your username and password.", "danger")
            
    return render_template('admin_login.html')


@app.route('/admin/logout')
def admin_logout():
    """Sign out administrator session."""
    session.pop('admin_logged_in', None)
    session.pop('admin_username', None)
    flash("You have been signed out from the administrator portal.", "info")
    return redirect(url_for('index'))


@app.route('/admin')
@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    """Admin Pharmacological Inventory & Visual Surveillance Dashboard."""
    # Metric counts
    total_drugs = query_db("SELECT COUNT(*) as cnt FROM drugs", one=True)['cnt']
    total_categories = query_db("SELECT COUNT(DISTINCT category) as cnt FROM drugs", one=True)['cnt']
    total_preds = query_db("SELECT COUNT(*) as cnt FROM predictions", one=True)['cnt']
    high_risk_preds = query_db("SELECT COUNT(*) as cnt FROM predictions WHERE predicted_risk = 'High'", one=True)['cnt']
    
    # Category distribution for bar chart
    category_counts = query_db("""
        SELECT category, COUNT(*) as count 
        FROM drugs 
        GROUP BY category 
        ORDER BY count DESC 
        LIMIT 10
    """)
    
    # Risk breakdown for donut chart
    risk_stats = {
        'low': query_db("SELECT COUNT(*) as cnt FROM predictions WHERE predicted_risk = 'Low'", one=True)['cnt'],
        'moderate': query_db("SELECT COUNT(*) as cnt FROM predictions WHERE predicted_risk = 'Moderate'", one=True)['cnt'],
        'high': query_db("SELECT COUNT(*) as cnt FROM predictions WHERE predicted_risk = 'High'", one=True)['cnt']
    }
    
    # Top clinically assessed drugs
    top_assessed = query_db("""
        SELECT drug_name, COUNT(*) as count 
        FROM predictions 
        GROUP BY drug_name 
        ORDER BY count DESC 
        LIMIT 6
    """)
    
    # Search and category filtering for inventory table
    q = request.args.get('q', '').strip()
    category_filter = request.args.get('category', '').strip()
    
    query_sql = "SELECT id, drug_name, generic_name, category, uses, common_side_effects, created_at FROM drugs WHERE 1=1"
    params = []
    if q:
        query_sql += " AND (drug_name LIKE ? OR generic_name LIKE ? OR uses LIKE ?)"
        params.extend([f"%{q}%", f"%{q}%", f"%{q}%"])
    if category_filter:
        query_sql += " AND category = ?"
        params.append(category_filter)
        
    query_sql += " ORDER BY id DESC LIMIT 60"
    drugs = query_db(query_sql, params)
    
    # Distinct categories for filter selector
    distinct_categories_rows = query_db("SELECT DISTINCT category FROM drugs ORDER BY category ASC LIMIT 50")
    distinct_categories = [r['category'] for r in distinct_categories_rows if r['category']]
    
    return render_template(
        'admin_dashboard.html',
        total_drugs=total_drugs,
        total_categories=total_categories,
        total_preds=total_preds,
        high_risk_preds=high_risk_preds,
        category_counts=category_counts,
        risk_stats=risk_stats,
        top_assessed=top_assessed,
        drugs=drugs,
        search_query=q,
        category_filter=category_filter,
        distinct_categories=distinct_categories
    )


@app.route('/admin/drug/add', methods=['POST'])
@admin_required
def admin_add_drug():
    """Add a new drug monograph to catalog."""
    drug_name = request.form.get('drug_name', '').strip()
    generic_name = request.form.get('generic_name', '').strip()
    category = request.form.get('category', '').strip()
    uses = request.form.get('uses', '').strip()
    common_side_effects = request.form.get('common_side_effects', '').strip()
    serious_warnings = request.form.get('serious_warnings', '').strip() or 'No specific black-box warnings reported at standard therapeutic levels.'
    contraindications = request.form.get('contraindications', '').strip() or 'Known hypersensitivity to active substance.'
    interactions = request.form.get('interactions', '').strip() or 'Consult pharmacology reference for comprehensive interaction profile.'
    precautions = request.form.get('precautions', '').strip() or 'Monitor clinical vitals and biochemical response periodically.'
    source_reference = request.form.get('source_reference', '').strip() or 'FDA / DailyMed / Clinical Compendium'
    
    if not (drug_name and generic_name and category and uses and common_side_effects):
        flash("Please complete all required fields (Drug Name, Generic Name, Category, Uses, Common Side Effects).", "danger")
        return redirect(url_for('admin_dashboard') + '#add-drug-section')
        
    # Check for existing duplicate drug name
    existing = query_db("SELECT id FROM drugs WHERE LOWER(drug_name) = LOWER(?)", (drug_name,), one=True)
    if existing:
        flash(f"A medication named '{drug_name}' already exists in catalog (ID: #{existing['id']}).", "warning")
        return redirect(url_for('admin_dashboard') + '#catalog-inventory')
        
    try:
        new_id = execute_db("""
            INSERT INTO drugs (drug_name, generic_name, category, uses, common_side_effects, serious_warnings, contraindications, interactions, precautions, source_reference)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (drug_name, generic_name, category, uses, common_side_effects, serious_warnings, contraindications, interactions, precautions, source_reference))
        
        flash(f"Medication '{drug_name}' (ID: #{new_id}) successfully added to database!", "success")
    except Exception as e:
        flash(f"Failed to add medication: {str(e)}", "danger")
        
    return redirect(url_for('admin_dashboard') + '#catalog-inventory')


@app.route('/admin/drug/delete/<int:drug_id>', methods=['POST'])
@admin_required
def admin_delete_drug(drug_id):
    """Delete a medication from catalog."""
    drug = query_db("SELECT drug_name FROM drugs WHERE id = ?", (drug_id,), one=True)
    if not drug:
        flash("Medication not found in catalog.", "warning")
        return redirect(url_for('admin_dashboard'))
        
    try:
        execute_db("DELETE FROM drugs WHERE id = ?", (drug_id,))
        flash(f"Medication '{drug['drug_name']}' (ID: #{drug_id}) removed from database.", "info")
    except Exception as e:
        flash(f"Failed to delete medication: {str(e)}", "danger")
        
    return redirect(url_for('admin_dashboard') + '#catalog-inventory')


# -------------------------------------------------------------
# REST & AJAX APIs
# -------------------------------------------------------------

@app.route('/api/ocr-scan', methods=['POST'])
def api_ocr_scan():
    """Handles async image upload, OpenCV preprocessing, and Tesseract OCR fuzzy matching."""
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file uploaded.'}), 400
        
    file_obj = request.files['image']
    if file_obj.filename == '':
        return jsonify({'success': False, 'error': 'Empty filename.'}), 400
        
    try:
        save_path, filename = validate_and_save_image(file_obj, app.config['UPLOAD_FOLDER'])
        
        # Extract text via OCR
        extracted_text = extract_text_from_image(save_path)
        
        # Query known drugs from database
        all_drugs = query_db("SELECT id, drug_name, generic_name, category FROM drugs")
        drug_list = [dict(d) for d in all_drugs]
        
        match_result = match_drug_from_text(extracted_text, drug_list)
        
        # Log OCR attempt
        execute_db("""
            INSERT INTO ocr_logs (image_filename, extracted_text, matched_drug_name, confidence_score)
            VALUES (?, ?, ?, ?)
        """, (
            filename,
            extracted_text,
            match_result['drug']['drug_name'] if match_result['drug'] else None,
            match_result['confidence']
        ))
        
        return jsonify({
            'success': True,
            'image_url': url_for('static', filename=f'uploads/{filename}'),
            'extracted_text': extracted_text,
            'identified': match_result['identified'],
            'confidence': match_result['confidence'],
            'matched_drug': match_result['drug'],
            'message': match_result['message']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/drugs/search')
def api_search_drugs():
    """Live search endpoint for drug autocomplete."""
    q = request.args.get('q', '').strip()
    if not q:
        drugs = query_db("SELECT id, drug_name, generic_name, category FROM drugs LIMIT 15")
    else:
        drugs = query_db(
            "SELECT id, drug_name, generic_name, category FROM drugs WHERE drug_name LIKE ? OR generic_name LIKE ? LIMIT 10",
            (f"%{q}%", f"%{q}%")
        )
    return jsonify([dict(d) for d in drugs])


@app.route('/api/predict', methods=['POST'])
def api_predict_json():
    """Programmatic prediction API."""
    data = request.get_json() or {}
    patient = data.get('patient', {})
    drug_id = data.get('drug_id')
    dosage_category = data.get('dosage_category', 'Standard')
    
    if not drug_id:
        return jsonify({'error': 'drug_id is required'}), 400
        
    drug_record = query_db("SELECT * FROM drugs WHERE id = ?", (drug_id,), one=True)
    if not drug_record:
        return jsonify({'error': 'Drug not found'}), 404
        
    try:
        res = predict_side_effect_risk(patient, dict(drug_record), dosage_category)
        return jsonify(res)
    except Exception as e:
        return jsonify({'error': str(e)}), 500



# -------------------------------------------------------------
# Error Handlers
# -------------------------------------------------------------

@app.errorhandler(404)
def not_found_error(error):
    return render_template('base.html', custom_body="<div class='container py-5 text-center'><h2>404 - Page Not Found</h2><p>The requested page does not exist.</p><a href='/' class='btn btn-primary'>Return to Home</a></div>"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('base.html', custom_body="<div class='container py-5 text-center'><h2>500 - Internal Server Error</h2><p>An unexpected server error occurred.</p><a href='/' class='btn btn-primary'>Return to Home</a></div>"), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"[*] Starting PharmacoRisk AI Web Server on http://127.0.0.1:{port} ...")
    app.run(host='0.0.0.0', port=port, debug=True)
