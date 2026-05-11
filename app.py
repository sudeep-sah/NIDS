import os
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash

# Import model-specific apps
from model_handlers.iot23_handler import IoT23Handler
from model_handlers.toniot_handler import TONIOTHandler
from model_handlers.unsw_handler import UNSWHandler
from model_handlers.cicids_handler import CICIDSHandler
from model_handlers.kdd_handler import KDDHandler

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-this-in-production'

# Initialize model handlers
model_handlers = {
    'iot23': IoT23Handler(),
    'toniot': TONIOTHandler(),
    'unsw_nb15': UNSWHandler(),
    'cicids': CICIDSHandler(),
    'kdd': KDDHandler()
}

# Database setup
def init_db():
    """
    Initialize the SQLite database and make sure all expected columns exist.
    
    This is written to be **backwards compatible** with older versions of the
    database that might be missing newer columns like `is_mitigated` or `user_id`.
    """
    conn = sqlite3.connect('ids_database.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    # Predictions table (base definition)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            model_name TEXT,
            input_features TEXT,
            prediction_result TEXT,
            confidence REAL,
            is_adversarial BOOLEAN,
            is_mitigated BOOLEAN DEFAULT 0,
            mitigation_suggestion TEXT,
            user_id INTEGER,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # ---- Backward‑compatibility: ensure important columns exist ----
    cursor.execute("PRAGMA table_info(predictions)")
    existing_cols = [row[1] for row in cursor.fetchall()]

    # Older DBs created without is_mitigated column will cause
    # "no such column: is_mitigated" during mitigation. Fix that here.
    if 'is_mitigated' not in existing_cols:
        cursor.execute(
            "ALTER TABLE predictions "
            "ADD COLUMN is_mitigated BOOLEAN DEFAULT 0"
        )

    # Some very old versions might also miss user_id; keep insert/save working
    if 'user_id' not in existing_cols:
        cursor.execute(
            "ALTER TABLE predictions "
            "ADD COLUMN user_id INTEGER"
        )

    conn.commit()
    conn.close()
    
def save_prediction(model_name, input_features, prediction_result, confidence, is_adversarial, mitigation):
    """
    Save a prediction to the database using the **local system time** for the
    timestamp so that History/Filters work correctly with the user's clock.
    """
    conn = sqlite3.connect('ids_database.db')
    cursor = conn.cursor()

    # Use local time (with timezone) and store as a simple string
    # Example: "2025-12-15 23:51:30"
    local_now = datetime.now().astimezone()
    timestamp_str = local_now.strftime('%Y-%m-%d %H:%M:%S')

    user_id = session.get('user_id')
    cursor.execute('''
        INSERT INTO predictions 
        (timestamp, model_name, input_features, prediction_result, confidence, is_adversarial, mitigation_suggestion, user_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp_str,
        model_name,
        json.dumps(input_features),
        prediction_result,
        confidence,
        is_adversarial,
        mitigation,
        user_id
    ))
    conn.commit()
    conn.close()

# User management functions
def create_user(username, email, password):
    conn = sqlite3.connect('ids_database.db')
    cursor = conn.cursor()
    try:
        password_hash = generate_password_hash(password)
        cursor.execute('''
            INSERT INTO users (username, email, password_hash)
            VALUES (?, ?, ?)
        ''', (username, email, password_hash))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_user_by_username(username):
    conn = sqlite3.connect('ids_database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    return user

def get_user_predictions(user_id):
    conn = sqlite3.connect('ids_database.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT timestamp, model_name, prediction_result, is_adversarial, mitigation_suggestion 
        FROM predictions 
        WHERE user_id = ?
        ORDER BY timestamp DESC
    ''', (user_id,))
    predictions = cursor.fetchall()
    conn.close()
    return predictions

# Login required decorator
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function


# Mitigation suggestions
MITIGATION_MAP = {
            'Normal': 'No action required - traffic appears normal.',
            'Benign': 'No action required - traffic appears normal.',
            'Dos': 'Implement rate limiting, use DoS protection services, configure load balancers, monitor traffic patterns for anomalies.',
            'Probe': 'Configure firewall rules, implement intrusion detection systems, use port knocking, monitor for reconnaissance activities.',
            'R2L': 'Strengthen authentication mechanisms, implement account lockout policies, use multi-factor authentication, monitor for unauthorized access attempts.',
            'U2R': 'Implement privilege escalation monitoring, keep systems patched, use application whitelisting, conduct regular security audits.',
            'BruteForce': 'Implement account lockout policies, strengthen password requirements, use multi-factor authentication.',
            'DDoS': 'Implement rate limiting, use DDoS protection services, configure firewalls to block suspicious traffic.',
            'TCPDDOS': 'Configure TCP flood protection, implement SYN cookies, use DDoS mitigation services.',
            'UDPDDOS': 'Block unused UDP ports, implement UDP flood protection, use traffic filtering.',
            'SQLInjection': 'Use parameterized queries, input validation, web application firewalls.',
            'PortScan': 'Configure firewall rules, implement intrusion detection systems, use port knocking.',
            'CMD': 'Implement command injection protection, sanitize user inputs, use least privilege principles.',
            'Samba': 'Update Samba services, restrict SMB protocols, use network segmentation.',
            'VNC': 'Use VNC over SSH tunnels, implement strong authentication, restrict VNC access to specific IPs.',
            'Analysis': 'Monitor for unusual traffic patterns, implement behavioral analysis tools.',
            'Backdoor': 'Conduct regular system scans, monitor for unusual processes, use application whitelisting.',
            'Exploits': 'Keep systems patched, use exploit mitigation techniques (ASLR, DEP), implement application security.',
            'Fuzzers': 'Implement input validation, use web application firewalls, monitor for abnormal requests.',
            'Reconnaissance': 'Limit information disclosure, monitor scan attempts, use honeypots.',
            'Shellcode': 'Implement memory protection mechanisms (ASLR, DEP), use antivirus software.',
            'Worms': 'Keep systems updated, use network segmentation, implement proper firewall rules.',
            'injection': 'Sanitize all user inputs, use prepared statements, implement content security policies.',
            'mitm': 'Use encrypted communications (HTTPS, VPN), implement certificate pinning, use secure protocols.',
            'password': 'Enforce strong password policies, implement multi-factor authentication, monitor for credential stuffing.',
            'ransomware': 'Maintain regular backups, use endpoint protection, educate users about phishing.',
            'scanning': 'Implement port security, use intrusion prevention systems, monitor for scan patterns.',
            'xss': 'Implement content security policy, sanitize user inputs, use XSS filters.',
            'Generic': 'General security best practices: update systems, monitor logs, implement defense in depth.'
        }
def get_mitigation_suggestion(attack_type):
    for key, value in MITIGATION_MAP.items():
        if key.lower() in attack_type.lower():
            return value
    return MITIGATION_MAP['Generic']

# Routes
@app.route('/')
def landing():
    """Landing page with project description"""
    return render_template('landing.html')

@app.route('/home')
@login_required
def index():
    """Main application page"""
    return render_template('index.html', models=list(model_handlers.keys()), username=session.get('username'))

@app.route('/about')
def about():
    return render_template('about.html', models=list(model_handlers.keys()))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not username or not email or not password:
            flash('All fields are required', 'error')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'error')
            return render_template('register.html')
        
        if create_user(username, email, password):
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Username or email already exists', 'error')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = get_user_by_username(username)
        if user and check_password_hash(user[3], password):  # user[3] is password_hash
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['email'] = user[2]
            flash('Login successful!', 'success')
            
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('landing'))





@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_csv():
    if request.method == 'POST':
        try:
            if 'csv_file' not in request.files:
                return render_template('upload.html', error='No file selected')
            
            file = request.files['csv_file']
            model_name = request.form.get('model_name')
            
            if file.filename == '':
                return render_template('upload.html', error='No file selected')
            
            if file and file.filename.endswith('.csv'):
                if model_name not in model_handlers:
                    return render_template('upload.html', error='Invalid model selected')
                print("model_name==",model_name)
            
                
                # Read CSV file
                if model_name=="unsw_nb15":
                    start_index = 47910   # zero-based index of the first *data* row you want
                    rows_to_read = 20
                    # if `file` is a Flask/Werkzeug upload, reset the pointer first
                    if hasattr(file, "stream"):
                        file.stream.seek(0)
                        df = pd.read_csv(file,skiprows=range(1, start_index + 1),nrows=rows_to_read)

                else:
                    df = pd.read_csv(file, nrows=20)
                
                # Get handler for the selected model
                handler = model_handlers[model_name]
                
                # Process each row and make predictions
                results = []
                shap_plots = []
                
                for index, row in df.iterrows():
                    try:
                        # Convert row to form data format
                        form_data = {str(col): str(val) for col, val in row.items()}
                        form_data['model_name'] = model_name
                        
                        # Make prediction
                        result = handler.predict(form_data)
                        
                        if 'error' not in result:
                            # Save to database
                            save_prediction(
                                model_name=model_name,
                                input_features=result.get('input_values', {}),
                                prediction_result=result['prediction'],
                                confidence=result.get('confidence', 0.0),
                                is_adversarial=False,  # CSV uploads are not adversarial
                                mitigation=get_mitigation_suggestion(result['prediction'])
                            )
                            
                            # FIXED: Use Python's 'in' operator instead of 'includes'
                            prediction_lower = result['prediction'].lower()
                            is_attack = 'normal' not in prediction_lower and 'benign' not in prediction_lower
                            
                            results.append({
                                'row_index': index + 1,
                                'prediction': result['prediction'],
                                'confidence': result.get('confidence', 0),
                                'is_attack': is_attack,
                                'shap_plot': result.get('shap_plot')
                            })
                            
                            # Collect unique SHAP plots
                            if result.get('shap_plot') and result.get('shap_plot') not in [p['path'] for p in shap_plots]:
                                shap_plots.append({
                                    'path': result.get('shap_plot'),
                                    'row_index': index + 1
                                })
                    
                    except Exception as e:
                        print(f"Error processing row {index}: {e}")
                        results.append({
                            'row_index': index + 1,
                            'prediction': f'Error: {str(e)}',
                            'confidence': 0,
                            'is_attack': False,
                            'error': True
                        })
                
                # Calculate statistics
                total_rows = len(results)
                attack_count = sum(1 for r in results if r.get('is_attack') and not r.get('error'))
                normal_count = total_rows - attack_count
                
                return render_template('upload_results.html',
                                    results=results,
                                    shap_plots=shap_plots,
                                    total_rows=total_rows,
                                    attack_count=attack_count,
                                    normal_count=normal_count,
                                    model_name=model_name,
                                    filename=file.filename)
            
            else:
                return render_template('upload.html', error='Please upload a CSV file')
                
        except Exception as e:
            return render_template('upload.html', error=f'Error processing file: {str(e)}')
    
    return render_template('upload.html', models=list(model_handlers.keys()))

# Add this route for individual row details
@app.route('/row_details/<int:row_index>')
def row_details(row_index):
    # This would need additional implementation to store and retrieve individual row data
    # For now, return a placeholder
    return jsonify({'message': f'Details for row {row_index} would be shown here'})
@app.route('/history')
@login_required
def history():
    conn = sqlite3.connect('ids_database.db')
    cursor = conn.cursor()
    
    # Get recent predictions for current user only
    user_id = session.get('user_id')
    cursor.execute('''
        SELECT timestamp, model_name, prediction_result, is_adversarial, mitigation_suggestion 
        FROM predictions 
        WHERE user_id = ?
        ORDER BY timestamp DESC 
    ''', (user_id,))
    recent_predictions = cursor.fetchall()
    
    # Get attack statistics for current user
    cursor.execute('''
        SELECT prediction_result, COUNT(*) 
        FROM predictions 
        WHERE user_id = ?
        GROUP BY prediction_result
    ''', (user_id,))
    attack_stats = cursor.fetchall()
    
    # Calculate stats for the cards
    total_predictions = sum(count for _, count in attack_stats)
    
    # Count Normal and Benign traffic
    normal_count = next((count for result, count in attack_stats if result.lower() == "normal"), 0)
    benign_count = next((count for result, count in attack_stats if result.lower() == "benign"), 0)
    normal_benign_count = normal_count + benign_count
    
    # Count adversarial attacks for current user
    cursor.execute('SELECT COUNT(*) FROM predictions WHERE is_adversarial = 1 AND user_id = ?', (user_id,))
    adversarial_count = cursor.fetchone()[0]
    
    available_models = list(set(prediction[1] for prediction in recent_predictions))
    available_attack_types = list(set(prediction[2] for prediction in recent_predictions if prediction[2].lower() not in ['normal', 'benign']))

    conn.close()
    
    return render_template('history.html', 
                         recent_predictions=recent_predictions,
                         attack_stats=attack_stats,
                         total_predictions=total_predictions,
                         normal_benign_count=normal_benign_count,
                         attack_count=total_predictions - normal_benign_count,
                         adversarial_count=adversarial_count,
                         available_models=available_models,
                         available_attack_types=available_attack_types,
                         username=session.get('username'))


@app.route('/predict', methods=['GET','POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            model_name = request.form.get('model_name')
            if model_name not in model_handlers:
                return jsonify({'error': 'Invalid model selected'}), 400

            handler = model_handlers[model_name]
            
            # Get prediction
            result = handler.predict(request.form)
            print("result==",result)
            
            if 'error' in result:
                return jsonify(result), 400

            # Save to database
            save_prediction(
                model_name=model_name,
                input_features=result.get('input_values', {}),
                prediction_result=result['prediction'],
                confidence=result.get('confidence', 0.0),
                is_adversarial=result.get('is_adversarial', False),
                mitigation=get_mitigation_suggestion(result['prediction'])
            )

            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    return render_template('detection.html', models=list(model_handlers.keys()))

@app.route('/dashboard')
@login_required
def dashboard():
    conn = sqlite3.connect('ids_database.db')
    cursor = conn.cursor()
    
    # Get recent predictions for current user only
    user_id = session.get('user_id')
    cursor.execute('''
        SELECT id, timestamp, model_name, prediction_result, is_adversarial, is_mitigated 
        FROM predictions 
        WHERE user_id = ?
        ORDER BY timestamp DESC 
        LIMIT 50
    ''', (user_id,))
    recent_predictions = cursor.fetchall()
    
    # Get attack statistics for current user
    cursor.execute('''
        SELECT prediction_result, COUNT(*) 
        FROM predictions 
        WHERE user_id = ?
        GROUP BY prediction_result
    ''', (user_id,))
    attack_stats = cursor.fetchall()
    
    # Calculate stats for the cards
    total_predictions = sum(count for _, count in attack_stats)
    
    # Count Normal and Benign traffic
    normal_count = next((count for result, count in attack_stats if result.lower() == "normal"), 0)
    benign_count = next((count for result, count in attack_stats if result.lower() == "benign"), 0)
    normal_benign_count = normal_count + benign_count
    
    # Count adversarial attacks for current user
    cursor.execute('SELECT COUNT(*) FROM predictions WHERE is_adversarial = 1 AND user_id = ?', (user_id,))
    adversarial_count = cursor.fetchone()[0]
    
    conn.close()
    
    return render_template('dashboard.html', 
                        recent_predictions=recent_predictions,
                         attack_stats=attack_stats,
                         total_predictions=total_predictions,
                         normal_benign_count=normal_benign_count,
                         attack_count=total_predictions - normal_benign_count,
                         adversarial_count=adversarial_count,
                         username=session.get('username'))
@app.route('/api/mitigate/<int:prediction_id>', methods=['POST'])
def mitigate_prediction(prediction_id):
    try:
        conn = sqlite3.connect('ids_database.db')
        cursor = conn.cursor()
        
        # First, check if the prediction exists
        cursor.execute('SELECT id FROM predictions WHERE id = ?', (prediction_id,))
        prediction = cursor.fetchone()
        
        if not prediction:
            conn.close()
            return jsonify({'success': False, 'error': 'Prediction not found'}), 404
        
        # Update the prediction as mitigated
        cursor.execute('''
            UPDATE predictions 
            SET is_mitigated = 1 
            WHERE id = ?
        ''', (prediction_id,))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Mitigation applied successfully'})
    
    except Exception as e:
        print(f"Mitigation error: {str(e)}")  # For server logs
        return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/api/predictions')
def get_predictions():
    conn = sqlite3.connect('ids_database.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT timestamp, model_name, prediction_result, is_adversarial 
        FROM predictions 
        ORDER BY timestamp DESC 
        LIMIT 100
    ''')
    predictions = cursor.fetchall()
    
    conn.close()
    
    return jsonify([{
        'timestamp': pred[0],
        'model_name': pred[1],
        'prediction_result': pred[2],
        'is_adversarial': bool(pred[3])
    } for pred in predictions])

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
