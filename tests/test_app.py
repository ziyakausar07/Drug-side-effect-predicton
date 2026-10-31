"""
Comprehensive Automated Test Suite for PharmacoRisk AI
Tests Web Routes, Clinical Prediction Flow, Input Validation, APIs, and Privacy Safeguards.
"""

import os
import json
import pytest
from app import app
from database.db import query_db, execute_db
from utils.prediction import predict_side_effect_risk, load_artifacts


@pytest.fixture
def client():
    """Create Flask test client with test configuration."""
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    
    with app.test_client() as test_client:
        with app.app_context():
            yield test_client


# -------------------------------------------------------------
# 1. Core Web Route Tests
# -------------------------------------------------------------

def test_home_page(client):
    """Test that home landing page loads successfully with required elements."""
    response = client.get('/')
    assert response.status_code == 200
    html = response.data.decode('utf-8')
    assert "PharmacoRisk AI" in html
    assert "Drug Directory" in html
    assert "Start Assessment" in html


def test_patient_form_get(client):
    """Test GET /patient renders demographics, conditions, and vitals fields."""
    response = client.get('/patient')
    assert response.status_code == 200
    html = response.data.decode('utf-8')
    assert "Patient Clinical Parameters" in html
    assert "Age (years)" in html
    assert "Systolic BP" in html
    assert "Medical Condition" in html
    assert "Allergy History" in html


def test_patient_form_validation_invalid_age(client):
    """Test POST /patient rejects invalid age (e.g. 150)."""
    response = client.post('/patient', data={
        'age': 150,
        'gender': 'Male',
        'weight': 70.0,
        'medical_condition': 'Hypertension',
        'allergy_history': 'None',
        'previous_adr': 'No',
        'current_medication': 'None',
        'systolic_bp': 120,
        'diastolic_bp': 80,
        'blood_sugar': 100.0,
        'creatinine': 1.0,
        'alt_enzyme': 25.0
    }, follow_redirects=True)
    assert response.status_code == 200
    html = response.data.decode('utf-8')
    assert "valid age between 1 and 120" in html


def test_patient_form_validation_invalid_bp(client):
    """Test POST /patient rejects invalid blood pressure."""
    response = client.post('/patient', data={
        'age': 45,
        'gender': 'Male',
        'weight': 70.0,
        'medical_condition': 'None',
        'allergy_history': 'None',
        'previous_adr': 'No',
        'current_medication': 'None',
        'systolic_bp': 300,  # Invalid
        'diastolic_bp': 80,
        'blood_sugar': 100.0,
        'creatinine': 1.0,
        'alt_enzyme': 25.0
    }, follow_redirects=True)
    assert response.status_code == 200
    html = response.data.decode('utf-8')
    assert "valid blood pressure" in html


def test_patient_form_valid_submission(client):
    """Test POST /patient accepts valid parameters and redirects to /drug-upload."""
    response = client.post('/patient', data={
        'age': 52,
        'gender': 'Female',
        'weight': 68.5,
        'medical_condition': 'Type 2 Diabetes',
        'allergy_history': 'Penicillin',
        'previous_adr': 'Yes',
        'current_medication': 'Metformin',
        'systolic_bp': 135,
        'diastolic_bp': 88,
        'blood_sugar': 140.0,
        'creatinine': 1.1,
        'alt_enzyme': 32.0
    }, follow_redirects=False)
    
    # Should redirect to drug-upload
    assert response.status_code == 302
    assert '/drug-upload' in response.headers['Location']


# -------------------------------------------------------------
# 2. Drug Selection, OCR, and Prediction Workflow
# -------------------------------------------------------------

def test_drug_upload_requires_patient_session(client):
    """Accessing /drug-upload without patient session should redirect to /patient."""
    with client.session_transaction() as sess:
        sess.clear()
        
    response = client.get('/drug-upload', follow_redirects=False)
    assert response.status_code == 302
    assert '/patient' in response.headers['Location']


def test_full_prediction_workflow(client):
    """Simulate a complete patient evaluation -> drug selection -> prediction view."""
    # 1. Submit patient details
    client.post('/patient', data={
        'age': 60,
        'gender': 'Male',
        'weight': 75.0,
        'medical_condition': 'Hypertension',
        'allergy_history': 'None',
        'previous_adr': 'No',
        'current_medication': 'None',
        'systolic_bp': 145,
        'diastolic_bp': 92,
        'blood_sugar': 110.0,
        'creatinine': 1.2,
        'alt_enzyme': 28.0
    }, follow_redirects=True)
    
    # 2. Get a valid drug from DB
    drug = query_db("SELECT id, drug_name FROM drugs LIMIT 1", one=True)
    assert drug is not None
    drug_id = drug['id']
    
    # 3. Post drug selection to run prediction
    response = client.post('/drug-upload', data={
        'drug_id': drug_id,
        'dosage_category': 'Standard'
    }, follow_redirects=True)
    
    assert response.status_code == 200
    html = response.data.decode('utf-8')
    assert "Risk Prediction Result" in html
    assert "Probability Distribution" in html
    assert "Contributing Risk Factors" in html
    assert drug['drug_name'] in html


def test_prediction_view_page(client):
    """Test that an existing prediction record displays patient and drug monographs."""
    latest_pred = query_db("SELECT id, drug_name FROM predictions ORDER BY id DESC LIMIT 1", one=True)
    if latest_pred:
        response = client.get(f"/prediction/{latest_pred['id']}")
        assert response.status_code == 200
        html = response.data.decode('utf-8')
        assert latest_pred['drug_name'] in html
        assert "Patient Profile" in html
        assert "Risk Level" in html


def test_drug_info_page(client):
    """Test /drug-info/<id> returns complete safety monograph."""
    drug = query_db("SELECT id, drug_name, generic_name, uses FROM drugs LIMIT 1", one=True)
    assert drug is not None
    response = client.get(f"/drug-info/{drug['id']}")
    assert response.status_code == 200
    html = response.data.decode('utf-8')
    assert drug['drug_name'] in html
    assert "Pharmacological Monograph" in html or "Drug Safety Information" in html or "Uses" in html


def test_drug_navbar_search(client):
    """Test /drug-info/search redirects to drug detail page."""
    drug = query_db("SELECT drug_name FROM drugs WHERE drug_name LIKE 'Metformin%' LIMIT 1", one=True)
    if not drug:
        drug = query_db("SELECT drug_name FROM drugs LIMIT 1", one=True)
    
    response = client.get(f"/drug-info/search?q={drug['drug_name']}", follow_redirects=False)
    assert response.status_code == 302
    assert '/drug-info/' in response.headers['Location']


# -------------------------------------------------------------
# 3. Dashboard & ML Details Obfuscation
# -------------------------------------------------------------

def test_dashboard_clinical_view(client):
    """Test dashboard presents clinical metrics and does NOT expose internal ML dev stats."""
    response = client.get('/dashboard')
    assert response.status_code == 200
    html = response.data.decode('utf-8')
    assert "Clinical Safety Dashboard" in html
    assert "Risk Distribution" in html
    assert "Clinical Risk Stratification" in html
    assert "Recent Predictions" in html
    
    # Verify that raw ML development details are NOT exposed to regular users
    assert "ML Model Information" not in html
    assert "training_samples" not in html
    assert "features_count" not in html
    assert "cv_mean" not in html


def test_admin_requires_authentication(client):
    """Test that /admin redirects unauthenticated users to /admin/login."""
    response = client.get('/admin', follow_redirects=False)
    assert response.status_code == 302
    assert '/admin/login' in response.headers['Location']


def test_admin_login_page_renders(client):
    """Test GET /admin/login renders login form with credentials fields."""
    response = client.get('/admin/login')
    assert response.status_code == 200
    html = response.data.decode('utf-8')
    assert "Administrator Access" in html
    assert "adminUsername" in html
    assert "adminPassword" in html


def test_admin_login_invalid_credentials(client):
    """Test POST /admin/login rejects invalid credentials."""
    response = client.post('/admin/login', data={
        'username': 'wronguser',
        'password': 'badpassword'
    }, follow_redirects=True)
    assert response.status_code == 200
    html = response.data.decode('utf-8')
    assert "Invalid administrator credentials" in html


def test_admin_login_and_dashboard_access(client):
    """Test administrator login success and dashboard surveillance graphs."""
    login_resp = client.post('/admin/login', data={
        'username': 'admin',
        'password': 'admin123'
    }, follow_redirects=True)
    assert login_resp.status_code == 200
    html = login_resp.data.decode('utf-8')
    assert "Drug Catalog & Safety Surveillance" in html
    assert "Total Drugs" in html
    assert "categoryBarChart" in html
    assert "riskDonutChart" in html


def test_admin_add_and_delete_drug(client):
    """Test admin adding a new drug to catalog and deleting it."""
    # 1. Login as admin
    client.post('/admin/login', data={'username': 'admin', 'password': 'admin123'})
    
    # 2. Add new test drug
    test_drug_name = "TestCure-Alpha-99"
    add_resp = client.post('/admin/drug/add', data={
        'drug_name': test_drug_name,
        'generic_name': 'Testcurium',
        'category': 'Novel Therapeutic Agent',
        'uses': 'Experimental clinical trial usage.',
        'common_side_effects': 'Mild headache, dry mouth.',
        'serious_warnings': 'None.',
        'contraindications': 'Hypersensitivity.',
        'interactions': 'None known.',
        'precautions': 'Routine monitoring.',
        'source_reference': 'Clinical Study Protocol 2026'
    }, follow_redirects=True)
    assert add_resp.status_code == 200
    
    # Verify drug exists in DB
    new_drug = query_db("SELECT id, drug_name FROM drugs WHERE drug_name = ?", (test_drug_name,), one=True)
    assert new_drug is not None
    assert new_drug['drug_name'] == test_drug_name
    
    # 3. Delete drug
    del_resp = client.post(f"/admin/drug/delete/{new_drug['id']}", follow_redirects=True)
    assert del_resp.status_code == 200
    
    # Verify drug removed from DB
    check_deleted = query_db("SELECT id FROM drugs WHERE drug_name = ?", (test_drug_name,), one=True)
    assert check_deleted is None


def test_admin_logout(client):
    """Test admin session logout."""
    client.post('/admin/login', data={'username': 'admin', 'password': 'admin123'})
    logout_resp = client.get('/admin/logout', follow_redirects=False)
    assert logout_resp.status_code == 302
    
    # Subsequent access to admin requires login again
    follow_resp = client.get('/admin', follow_redirects=False)
    assert follow_resp.status_code == 302
    assert '/admin/login' in follow_resp.headers['Location']


# -------------------------------------------------------------
# 4. REST APIs
# -------------------------------------------------------------

def test_api_search_drugs(client):
    """Test live autocomplete API."""
    response = client.get('/api/drugs/search?q=aspirin')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)


def test_api_predict_json(client):
    """Test programmatic prediction endpoint."""
    drug = query_db("SELECT id FROM drugs LIMIT 1", one=True)
    assert drug is not None
    
    payload = {
        'drug_id': drug['id'],
        'dosage_category': 'Standard',
        'patient': {
            'age': 50,
            'gender': 'Male',
            'weight': 75.0,
            'medical_condition': 'Hypertension',
            'allergy_history': 'None',
            'previous_adr': 'No',
            'current_medication': 'None',
            'systolic_bp': 130,
            'diastolic_bp': 85,
            'blood_sugar': 105.0,
            'creatinine': 1.0,
            'alt_enzyme': 24.0
        }
    }
    
    response = client.post('/api/predict', json=payload)
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'predicted_risk' in data
    assert data['predicted_risk'] in ['Low', 'Moderate', 'High']
    assert 'probabilities' in data
    assert 'contributing_factors' in data


# -------------------------------------------------------------
# 5. Direct ML Engine Function Unit Tests
# -------------------------------------------------------------

def test_ml_risk_engine():
    """Directly test predict_side_effect_risk with safe vs high-risk patient profile."""
    # Ensure artifacts are loaded
    load_artifacts()
    
    drug_mock = {
        'id': 99999,
        'drug_name': 'TestMed',
        'generic_name': 'TestGeneric',
        'category': 'Antibiotic',
        'uses': 'Bacterial infection',
        'common_side_effects': 'Nausea, headache',
        'serious_warnings': 'Severe renal injury',
        'contraindications': 'Renal impairment, hypersensitivity',
        'interactions': 'NSAIDs',
        'precautions': 'Monitor kidney function'
    }
    
    # 1. Healthy patient
    healthy_patient = {
        'age': 28,
        'gender': 'Female',
        'weight': 60.0,
        'medical_condition': 'None',
        'allergy_history': 'None',
        'previous_adr': 'No',
        'current_medication': 'None',
        'systolic_bp': 118,
        'diastolic_bp': 78,
        'blood_sugar': 90.0,
        'creatinine': 0.8,
        'alt_enzyme': 20.0
    }
    healthy_res = predict_side_effect_risk(healthy_patient, drug_mock, 'Low')
    assert healthy_res['predicted_risk'] in ['Low', 'Moderate', 'High']
    assert isinstance(healthy_res['probabilities'], dict)
    assert 'Low' in healthy_res['probabilities']
    
    # 2. High risk patient (severe renal impairment, high ALT, previous ADR, high dose)
    critical_patient = {
        'age': 78,
        'gender': 'Male',
        'weight': 50.0,
        'medical_condition': 'Chronic Kidney Disease',
        'allergy_history': 'Penicillin',
        'previous_adr': 'Yes',
        'current_medication': 'NSAIDs (Ibuprofen / Diclofenac / Naproxen)',
        'systolic_bp': 180,
        'diastolic_bp': 110,
        'blood_sugar': 250.0,
        'creatinine': 3.5,  # Very high creatinine
        'alt_enzyme': 120.0  # Elevated liver enzyme
    }
    critical_res = predict_side_effect_risk(critical_patient, drug_mock, 'High')
    assert critical_res['predicted_risk'] in ['Moderate', 'High']
    assert len(critical_res['contributing_factors']) > 0
