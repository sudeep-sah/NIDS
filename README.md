# 🛡️ AI-Powered Explainable Intrusion Detection System With Cross-Domain Adaptability

An advanced AI-driven Intrusion Detection System (IDS) designed to detect, explain, and mitigate cyber threats across IoT, cloud, and enterprise environments using Machine Learning, Deep Learning, Explainable AI (XAI), and adversarial defense mechanisms.

---

## 📌 Overview

Traditional Intrusion Detection Systems mainly rely on static signature-based detection methods that fail to identify unknown or evolving cyberattacks. This project introduces an intelligent and explainable IDS framework capable of:

- Detecting known and zero-day attacks
- Explaining prediction decisions using SHAP
- Defending against adversarial attacks
- Suggesting automated mitigation strategies
- Supporting cross-domain adaptability across multiple datasets

The system combines Machine Learning, Deep Learning, Explainable AI, and Flask-based deployment into a unified cybersecurity solution.

---

# 🚀 Key Features

✅ Hybrid ML + DL Intrusion Detection  
✅ Explainable AI using SHAP Visualization  
✅ Cross-Domain Adaptability  
✅ Adversarial Defense Mechanism  
✅ Automated Mitigation Suggestions  
✅ Flask Web Dashboard  
✅ SQLite Database Integration  
✅ Attack History & Filtering System  
✅ Real-Time Prediction Interface  
✅ Multi-Dataset Support

---

# 🧠 Technologies Used

## Programming Language
- Python

## Machine Learning & Deep Learning
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Random Forest
- CNN (Convolutional Neural Network)

## Explainable AI
- SHAP (SHapley Additive exPlanations)

## Web Framework
- Flask

## Database
- SQLite

## Frontend
- HTML
- CSS
- JavaScript

---

# 📂 Supported Datasets

The system is trained and evaluated on multiple benchmark cybersecurity datasets:

- CICIDS 2017
- TON_IoT
- IoT-23
- NSL-KDD
- UNSW-NB15

These datasets cover a wide range of network traffic behaviors and attack categories.

---

# 🏗️ System Architecture

The project workflow consists of:

1. Dataset Collection
2. Data Preprocessing
3. Feature Extraction & Selection
4. ML & DL Model Training
5. Intrusion Prediction
6. SHAP Explainability Analysis
7. Adversarial Defense
8. Mitigation Recommendation
9. Database Logging & Visualization

---

# ⚙️ Core Modules

## 🔹 Data Preprocessing
- Data Cleaning
- Missing Value Handling
- Label Encoding
- Normalization

## 🔹 Feature Selection
- Correlation Analysis
- Information Gain
- Dimensionality Reduction

## 🔹 Classification Models
- XGBoost
- Random Forest
- CNN

## 🔹 Explainability Layer
The SHAP framework is used to explain model predictions by visualizing the contribution of each feature toward attack detection.

## 🔹 Adversarial Defense
The system integrates adversarial training and noise injection techniques to improve robustness against evasion and poisoning attacks.

## 🔹 Mitigation Engine
Automatically recommends security countermeasures such as:
- IP Blocking
- Traffic Filtering
- Rate Limiting
- Multi-Factor Authentication
- SQL Injection Prevention

---

# 📊 Performance

The proposed IDS achieved:

- High Detection Accuracy (>97%)
- Reduced False Positives
- Improved Cross-Domain Adaptability
- Enhanced Model Transparency
- Robustness Against Adversarial Attacks

---

# 🖥️ Web Interface Features

- User Authentication
- Dataset Upload
- Intrusion Prediction
- SHAP Visualization
- Attack History
- Attack Filtering
- Dashboard Analytics
- Mitigation Suggestions

---

# 📁 Project Structure

```bash
NIDS/
│
├── model_handlers/
├── models/
├── static/
├── templates/
├── app.py
├── database.py
├── final.csv
├── NSL-KDD.csv
├── UNSW-NB15.csv
├── IoT-23.csv
└── ton-iot.csv
```

---

# ▶️ Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/sudeep-sah/NIDS.git
```

## 2️⃣ Navigate to Project Folder

```bash
cd NIDS
```

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 4️⃣ Run Application

```bash
python app.py
```

---

# 📌 Future Enhancements

- Real-Time Packet Monitoring
- Cloud Deployment
- Federated Learning
- Edge Computing Integration
- Automated Threat Response
- SDN Integration
- Reinforcement Learning Based Mitigation



---

# 📚 Research References

This project is inspired by modern research in:
- Explainable AI
- Adversarial Machine Learning
- Network Intrusion Detection Systems
- Cross-Domain Cybersecurity Frameworks

---

# 📜 License

This project is developed for educational and research purposes.
