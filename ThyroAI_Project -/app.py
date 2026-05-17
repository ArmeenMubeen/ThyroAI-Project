from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import joblib
import os
import sqlite3
import hashlib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ── Database Setup ───────────────────────────────────
def init_db():
    conn = sqlite3.connect('thyroai.db')
    c = conn.cursor()

    # Patients table
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        age INTEGER,
        gender TEXT,
        risk_level TEXT,
        detection_result TEXT,
        detection_confidence REAL,
        stage TEXT,
        stage_confidence REAL,
        metastasis TEXT,
        metastasis_risk REAL,
        overall_risk TEXT
    )''')

    # Doctors table
    c.execute('''CREATE TABLE IF NOT EXISTS doctors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        name TEXT
    )''')

    # Add default doctor account
    password = hashlib.sha256('thyroai2024'.encode()).hexdigest()
    try:
        c.execute("INSERT INTO doctors (username, password, name) VALUES (?,?,?)",
                  ('doctor', password, 'Dr. Admin'))
    except:
        pass

    conn.commit()
    conn.close()
    print("✅ Database initialized!")

init_db()

# ── Load Models ──────────────────────────────────────
print("\n📂 Checking model files:")
required_files = [
    'models/module1_detection_model.pkl',
    'models/module1_scaler.pkl',
    'models/module2_staging_model.pkl',
    'models/module2_scaler.pkl',
    'models/module2_label_encoder.pkl',
    'models/module3_metastasis_model.pkl',
    'models/module3_scaler.pkl',
    'models/module3_label_encoder.pkl',
    'models/feature_encoders.pkl',
]
for f in required_files:
    status = "✅" if os.path.exists(f) else "❌"
    print(f"  {status} {f}")

try:
    m1_model         = joblib.load('models/module1_detection_model.pkl')
    m1_scaler        = joblib.load('models/module1_scaler.pkl')
    m2_model         = joblib.load('models/module2_staging_model.pkl')
    m2_scaler        = joblib.load('models/module2_scaler.pkl')
    m2_enc           = joblib.load('models/module2_label_encoder.pkl')
    m3_model         = joblib.load('models/module3_metastasis_model.pkl')
    m3_scaler        = joblib.load('models/module3_scaler.pkl')
    m3_enc           = joblib.load('models/module3_label_encoder.pkl')

    # Load feature encoders — fallback if missing
    if os.path.exists('models/feature_encoders.pkl'):
        feature_encoders = joblib.load('models/feature_encoders.pkl')
        print("✅ Feature encoders loaded!")
    else:
        print("⚠️ feature_encoders.pkl missing — using fallback")
        feature_encoders = None

    print("\n✅ All models loaded successfully!")
except Exception as e:
    print(f"\n❌ Error: {e}")
    feature_encoders = None

M1_COLS = ['Age','Gender','Smoking','Hx Smoking','Hx Radiothreapy',
           'Thyroid Function','Physical Examination','Adenopathy',
           'Pathology','Focality','Risk','T','N','M','Stage','Response']
M2_COLS = ['Age','Gender','Smoking','Hx Smoking','Hx Radiothreapy',
           'Thyroid Function','Physical Examination','Adenopathy',
           'Pathology','Focality','Risk','T','N','M','Response','Recurred']
M3_COLS = ['Age','Gender','Smoking','Hx Smoking','Hx Radiothreapy',
           'Thyroid Function','Physical Examination','Adenopathy',
           'Pathology','Focality','Risk','T','N','Stage','Response','Recurred']

def prepare(data, cols):
    df = pd.DataFrame([data])

    # Add defaults
    if 'Recurred' not in df.columns:
        df['Recurred'] = 'No'
    if 'Stage' not in df.columns:
        df['Stage'] = 'I'
    if 'M' not in df.columns:
        df['M'] = 'M0'

    cat_cols = ['Gender','Smoking','Hx Smoking','Hx Radiothreapy',
                'Thyroid Function','Physical Examination','Adenopathy',
                'Pathology','Focality','Risk','T','N','M',
                'Stage','Response','Recurred']

    if feature_encoders is not None:
        # Use fixed encoders
        for col in cat_cols:
            if col in df.columns:
                try:
                    df[col] = feature_encoders[col].transform(
                        df[col].astype(str))
                except:
                    df[col] = 0
    else:
        # Fallback — basic encoding
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in cat_cols:
            if col in df.columns:
                df[col] = le.fit_transform(df[col].astype(str))

    return df[cols]

# ── Routes ───────────────────────────────────────────
@app.route('/')
def home():
    return jsonify({'message': 'ThyroAI API v2.0', 'status': 'running'})

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'version': '2.0'})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = hashlib.sha256(data.get('password','').encode()).hexdigest()
    conn = sqlite3.connect('thyroai.db')
    c = conn.cursor()
    c.execute("SELECT name FROM doctors WHERE username=? AND password=?",
              (username, password))
    user = c.fetchone()
    conn.close()
    if user:
        return jsonify({'status': 'success', 'name': user[0]})
    return jsonify({'status': 'error', 'message': 'Invalid credentials'}), 401

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({'status':'error','message':'No data'}), 400

        # Module 1
        df1   = prepare(data, M1_COLS)
        X1    = pd.DataFrame(m1_scaler.transform(df1), columns=df1.columns)
        pred1 = int(m1_model.predict(X1)[0])
        prob1 = float(m1_model.predict_proba(X1)[0][1])

        # Module 2
        df2   = prepare(data, M2_COLS)
        X2    = pd.DataFrame(m2_scaler.transform(df2), columns=df2.columns)
        pred2 = int(m2_model.predict(X2)[0])
        prob2 = float(max(m2_model.predict_proba(X2)[0]))
        stage = str(m2_enc.inverse_transform([pred2])[0])

        # Module 3
        df3   = prepare(data, M3_COLS)
        X3    = pd.DataFrame(m3_scaler.transform(df3), columns=df3.columns)
        pred3 = int(m3_model.predict(X3)[0])
        prob3 = float(m3_model.predict_proba(X3)[0][1])
        meta  = str(m3_enc.inverse_transform([pred3])[0])

        # Overall risk
        if pred1==1 or meta=='M1' or prob1>0.6:
            overall = 'HIGH'
        elif prob1>0.3:
            overall = 'MEDIUM'
        else:
            overall = 'LOW'

        # Save to database
        conn = sqlite3.connect('thyroai.db')
        c = conn.cursor()
        c.execute('''INSERT INTO predictions
            (timestamp,age,gender,risk_level,detection_result,
             detection_confidence,stage,stage_confidence,
             metastasis,metastasis_risk,overall_risk)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)''', (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            data.get('Age'), data.get('Gender'),
            data.get('Risk'),
            'Positive' if pred1==1 else 'Negative',
            round(prob1*100,2), stage, round(prob2*100,2),
            meta, round(prob3*100,2), overall))
        conn.commit()
        conn.close()

        return jsonify({
            'status'     : 'success',
            'patient'    : {'age': data.get('Age'), 'gender': data.get('Gender')},
            'detection'  : {'result': 'Positive' if pred1==1 else 'Negative',
                           'confidence': round(prob1*100,2)},
            'staging'    : {'stage': stage, 'confidence': round(prob2*100,2)},
            'metastasis' : {'status': meta, 'risk': round(prob3*100,2)},
            'overall_risk': overall
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status':'error','message':str(e)}), 400

@app.route('/history')
def history():
    conn = sqlite3.connect('thyroai.db')
    c = conn.cursor()
    c.execute('''SELECT timestamp,age,gender,detection_result,
                 stage,metastasis,overall_risk
                 FROM predictions
                 ORDER BY id DESC LIMIT 10''')
    rows = c.fetchall()
    conn.close()
    cols = ['timestamp','age','gender','detection',
            'stage','metastasis','overall_risk']
    return jsonify({'status':'success',
                    'history':[dict(zip(cols,r)) for r in rows]})

if __name__ == '__main__':
    print("\n🧬 Starting ThyroAI API v2.0...")
    app.run(debug=True, port=5000)