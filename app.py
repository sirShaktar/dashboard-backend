from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)
CORS(app)

# Ensure we get the absolute path to the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "report_236.csv")

def load_and_clean_data():
    try:
        if not os.path.exists(DATA_PATH):
            print(f"Data file not found at {DATA_PATH}")
            # For testing without real data, return dummy data if file is missing (though the prompt says NO FAKE DATA, I will stick to returning an error or dummy if strictly needed, but let's just return empty df). The prompt says: "Use my LOCAL dataset path in Python. Do NOT generate fake data."
            return pd.DataFrame()

        df = pd.read_csv(DATA_PATH)
        
        # Select required columns
        cols = ['q87_confidence', 'q95_cyber_awareness', 'q96digital_corruption']
        available_cols = [c for c in cols if c in df.columns]
        
        if not available_cols:
            print("Required columns not found")
            return pd.DataFrame()
            
        df = df[available_cols]
        
        # remove blank spaces
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()
            
        # convert to numeric
        for col in available_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # drop missing values
        df = df.dropna()
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

@app.route('/api/stats', methods=['GET'])
def get_stats():
    df = load_and_clean_data()
    if df.empty:
        return jsonify({"error": "Data not available"}), 500
        
    stats = {}
    for col in df.columns:
        stats[col] = {
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std())
        }
    return jsonify(stats)

@app.route('/api/distribution', methods=['GET'])
def get_distribution():
    df = load_and_clean_data()
    if df.empty:
        return jsonify({"error": "Data not available"}), 500
        
    dist = {}
    for col in df.columns:
        counts = df[col].value_counts().to_dict()
        dist[col] = [{"name": str(k), "count": int(v)} for k, v in counts.items()]
    return jsonify(dist)

@app.route('/api/relationship', methods=['GET'])
def get_relationship():
    df = load_and_clean_data()
    if df.empty:
        return jsonify({"error": "Data not available"}), 500
    
    if 'q95_cyber_awareness' in df.columns and 'q87_confidence' in df.columns:
        # Just return downsampled data if it's too large, but let's return up to 500 points for scatter plot
        data = df[['q95_cyber_awareness', 'q87_confidence']].head(500).to_dict(orient='records')
        return jsonify(data)
    return jsonify({"error": "Columns not found"}), 400

@app.route('/api/correlation', methods=['GET'])
def get_correlation():
    df = load_and_clean_data()
    if df.empty:
        return jsonify({"error": "Data not available"}), 500
        
    corr = df.corr().to_dict()
    return jsonify(corr)

@app.route('/api/model-results', methods=['GET'])
def get_model_results():
    df = load_and_clean_data()
    if df.empty:
        return jsonify({"error": "Data not available"}), 500
        
    # We will use q87_confidence as target
    if 'q87_confidence' not in df.columns:
         return jsonify({"error": "Target column missing"}), 400
         
    target_col = 'q87_confidence'
    feature_cols = [c for c in df.columns if c != target_col]
    
    if not feature_cols:
         return jsonify({"error": "Feature columns missing"}), 400
         
    X = df[feature_cols]
    y = df[target_col]
    
    # Treat as classification
    y = y.astype(str)
    
    # Check if we have more than 1 class
    if len(y.unique()) <= 1:
        return jsonify([
            {"name": "Logistic Regression", "accuracy": 1.0},
            {"name": "Decision Tree", "accuracy": 1.0},
            {"name": "Random Forest", "accuracy": 1.0}
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }
    
    results = []
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = float(accuracy_score(y_test, preds))
            results.append({"name": name, "accuracy": round(acc, 4)})
        except Exception as e:
            results.append({"name": name, "accuracy": 0.0, "error": str(e)})
            
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
