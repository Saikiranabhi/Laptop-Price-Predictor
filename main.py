# main.py
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import re

# ── Feature Engineering ──────────────────────────────────────────────────────

def clean_ram(x):        return int(re.search(r'(\d+)', str(x)).group(1))
def clean_weight(x):     return float(re.search(r'([\d.]+)', str(x)).group(1))

def extract_cpu(x):
    for b in ['Intel Core i7','Intel Core i5','Intel Core i3']:
        if b in str(x): return b
    return 'AMD' if 'AMD' in str(x) else 'Intel Other' if 'Intel' in str(x) else 'Other'

def extract_gpu(x):
    if 'Nvidia' in str(x) or 'nvidia' in str(x): return 'Nvidia'
    return 'AMD' if 'AMD' in str(x) else 'Intel' if 'Intel' in str(x) else 'Other'

def parse_memory(x):
    hdd, ssd = 0, 0
    m = re.search(r'(\d+)GB\s*(SSD|Flash)', str(x), re.IGNORECASE)
    if m: ssd = int(m.group(1))
    m = re.search(r'(\d+)GB\s*HDD', str(x), re.IGNORECASE)
    if m: hdd = int(m.group(1))
    m = re.search(r'(\d+)TB', str(x), re.IGNORECASE)
    if m:
        if 'SSD' in str(x) or 'Flash' in str(x): ssd = int(m.group(1))*1024
        else: hdd = int(m.group(1))*1024
    return hdd, ssd

def extract_res(x):
    m = re.search(r'(\d{3,4})x(\d{3,4})', str(x))
    return (int(m.group(1)), int(m.group(2))) if m else (1920, 1080)

def preprocess(df):
    df = df.copy()
    df['Ram']        = df['Ram'].apply(clean_ram)
    df['Weight']     = df['Weight'].apply(clean_weight)
    df['Cpu_brand']  = df['Cpu'].apply(extract_cpu)
    df['Gpu_brand']  = df['Gpu'].apply(extract_gpu)
    df[['HDD','SSD']]= df['Memory'].apply(lambda x: pd.Series(parse_memory(x)))
    df[['Xr','Yr']]  = df['ScreenResolution'].apply(lambda x: pd.Series(extract_res(x)))
    df['ppi']        = ((df['Xr']**2 + df['Yr']**2)**0.5) / df['Inches']
    df['Touchscreen']= df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in str(x) else 0)
    df['IPS']        = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in str(x) else 0)
    # df.drop(columns=['Cpu','Memory','ScreenResolution','Gpu','Xr','Yr'], inplace=True)
    drop_cols = [c for c in ['Unnamed: 0','Cpu','Memory','ScreenResolution','Gpu','Xr','Yr'] 
                 if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    le_map = {}
    for col in ['Company','TypeName','Cpu_brand','Gpu_brand','OpSys']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_map[col] = le

    return df, le_map

# ── Train ─────────────────────────────────────────────────────────────────────

def main():
    # Load data
    print("📂 Loading data...")
    # df = pd.read_excel('data/raw/laptop_data.xlsx')
    df = pd.read_excel('data/raw/laptop_data.xlsx', index_col=0)
    print(f"✅ Loaded {len(df)} rows")

    # Preprocess
    print("⚙️  Preprocessing...")
    df_processed, le_map = preprocess(df)

    # Features & target
    X = df_processed.drop(columns=['Price'])
    y = np.log(df_processed['Price'])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Random Forest
    print("🚀 Training Random Forest...")
    model = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"✅ R² Score: {score:.4f}")

    # Save model
    os.makedirs('models', exist_ok=True)
    # joblib.dump(model,  'models/best_model.pkl')
    # joblib.dump(le_map, 'models/label_encoders.pkl')
    # print("💾 Model saved to models/best_model.pkl")
    joblib.dump(model,            'models/best_model.pkl')
    joblib.dump(le_map,           'models/label_encoders.pkl')
    joblib.dump(list(X.columns),  'models/feature_cols.pkl')
    print(f"✅ Features: {list(X.columns)}")
    print("💾 Models saved to models/")
    print("🎉 Done! Now run: uv run streamlit run src/api/app.py")

if __name__ == "__main__":
    main()