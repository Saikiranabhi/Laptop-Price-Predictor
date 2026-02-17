# # # Auto-generated file
# # from src.data.load_data import DataLoader
# # from src.data.preprocess import FeatureEngineer
# # from src.models.train import ModelTrainer
# # import joblib

# # def main():
# #     # 1. Load Data
# #     loader = DataLoader('data/raw/laptop_data.xlsx')
# #     df = loader.load()
# #     loader.validate_schema(df)
    
# #     # 2. Feature Engineering
# #     fe = FeatureEngineer()
# #     df_processed = fe.fit_transform(df)
    
# #     # Save preprocessed data
# #     df_processed.to_csv('data/processed/preprocessed_data.csv', index=False)
    
# #     # Save feature engineer for inference
# #     joblib.dump(fe, 'models/feature_engineer.pkl')
    
# #     # 3. Train Models
# #     trainer = ModelTrainer()
# #     X_train, X_test, y_train, y_test = trainer.prepare_data(df_processed)
# #     trainer.train_all(X_train, y_train, X_test, y_test)
    
# #     # 4. Compare Models
# #     comparison_df = trainer.print_comparison_table()
# #     comparison_df.to_csv('models/model_comparison.csv')
    
# #     # 5. Save Best Model
# #     best_name, best_model, best_metrics = trainer.get_best_model()
# #     trainer.save_best_model(best_model)
    
# #     print(f"\n🎉 Training Complete! Best Model: {best_name}")
# #     print(f"   Accuracy Improvement: Check model_comparison.csv")
# #     print(f"   RMSE Reduction: {best_metrics['test_rmse']:.4f}")

# # if __name__ == "__main__":
# #     main()



# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd
# import re
# import os

# st.set_page_config(
#     page_title="Laptop Price Predictor",
#     page_icon="💻",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# st.markdown("""
# <style>
# body { background-color: #f0f2f6; }
# .prediction-box {
#     background: linear-gradient(135deg, #667eea, #764ba2);
#     padding: 30px; border-radius: 15px;
#     color: white; text-align: center;
#     box-shadow: 0 8px 24px rgba(0,0,0,0.2);
# }
# .stat-card {
#     background: white; padding: 20px;
#     border-radius: 10px; text-align: center;
#     box-shadow: 0 4px 12px rgba(0,0,0,0.1);
# }
# .stButton > button {
#     background: linear-gradient(135deg, #667eea, #764ba2);
#     color: white; font-size: 18px;
#     border-radius: 10px; border: none;
#     padding: 12px 30px; width: 100%;
#     font-weight: bold;
# }
# </style>
# """, unsafe_allow_html=True)


# # ─── Helpers ────────────────────────────────────────────────────────────────

# def clean_ram(ram_str):
#     return int(re.search(r'(\d+)', str(ram_str)).group(1))

# def clean_weight(weight_str):
#     return float(re.search(r'([\d.]+)', str(weight_str)).group(1))

# def extract_cpu_brand(cpu_str):
#     cpu_str = str(cpu_str)
#     for brand in ['Intel Core i7', 'Intel Core i5', 'Intel Core i3']:
#         if brand in cpu_str:
#             return brand
#     return 'AMD' if 'AMD' in cpu_str else 'Intel Other' if 'Intel' in cpu_str else 'Other'

# def extract_gpu_brand(gpu_str):
#     gpu_str = str(gpu_str)
#     if 'Nvidia' in gpu_str or 'nvidia' in gpu_str:
#         return 'Nvidia'
#     return 'AMD' if 'AMD' in gpu_str else 'Intel' if 'Intel' in gpu_str else 'Other'

# def parse_memory(memory_str):
#     hdd, ssd = 0, 0
#     m = re.search(r'(\d+)GB\s*(SSD|Flash)', memory_str, re.IGNORECASE)
#     if m:
#         ssd = int(m.group(1))
#     m = re.search(r'(\d+)GB\s*HDD', memory_str, re.IGNORECASE)
#     if m:
#         hdd = int(m.group(1))
#     m = re.search(r'(\d+)TB', memory_str, re.IGNORECASE)
#     if m:
#         if 'SSD' in memory_str or 'Flash' in memory_str:
#             ssd = int(m.group(1)) * 1024
#         else:
#             hdd = int(m.group(1)) * 1024
#     return hdd, ssd

# def calculate_ppi(x_res, y_res, inches):
#     return ((x_res**2 + y_res**2)**0.5) / inches if inches > 0 else 0


# # @st.cache_resource
# # def load_model():
# #     try:
# #         model = joblib.load('models/best_model.pkl')
# #         return model
# #     except Exception:
# #         st.warning("⚠️ Model file not found. Using demo mode.")
# #         return None

# # In app.py — replace load_model() with:
# @st.cache_resource
# def load_model():
#     # Walk up from src/api/ to project root
#     base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#     model_path = os.path.join(base, 'models', 'best_model.pkl')
    
#     if not os.path.exists(model_path):
#         st.error(f"❌ Model not found at {model_path}. Run `uv run python main.py` first.")
#         st.stop()
    
#     return joblib.load(model_path)

# @st.cache_data
# def load_data():
#     try:
#         return pd.read_excel('data/raw/laptop_data.xlsx')
#     except Exception:
#         return None


# # ─── Layout ─────────────────────────────────────────────────────────────────

# st.markdown("<h1 style='text-align:center; color:#667eea;'>💻 Laptop Price Predictor</h1>", unsafe_allow_html=True)
# st.markdown("<p style='text-align:center; color:#888;'>ML-powered price estimation using Random Forest & XGBoost</p>", unsafe_allow_html=True)
# st.markdown("---")

# model = load_model()

# # ─── Sidebar inputs ──────────────────────────────────────────────────────────

# with st.sidebar:
#     st.header("🔧 Configure Laptop")

#     company = st.selectbox('🏢 Brand', ['Apple', 'Dell', 'HP', 'Lenovo', 'Asus',
#                                          'Acer', 'MSI', 'Toshiba', 'Samsung', 'Razer'])
#     laptop_type = st.selectbox('📱 Type', ['Ultrabook', 'Notebook', 'Gaming',
#                                             'Workstation', '2 in 1 Convertible', 'Netbook'])
#     screen_size = st.slider('📏 Screen Size (inches)', 10.0, 18.0, 15.6, 0.1)
#     resolution = st.selectbox('🖥️ Resolution', [
#         '1920x1080', '1366x768', '2560x1440', '3840x2160',
#         '2560x1600', '3200x1800', '2880x1800', '2304x1440'
#     ])

#     col1, col2 = st.columns(2)
#     with col1:
#         touchscreen = st.checkbox('✋ Touchscreen')
#     with col2:
#         ips = st.checkbox('🎨 IPS')

#     cpu = st.selectbox('⚙️ CPU', ['Intel Core i3', 'Intel Core i5',
#                                     'Intel Core i7', 'AMD', 'Intel Other'])
#     ram = st.select_slider('🧠 RAM (GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64], value=8)

#     st.subheader("💾 Storage")
#     c1, c2 = st.columns(2)
#     with c1:
#         hdd = st.selectbox('HDD (GB)', [0, 128, 256, 512, 1024, 2048])
#     with c2:
#         ssd = st.selectbox('SSD (GB)', [0, 8, 128, 256, 512, 1024], index=3)

#     gpu = st.selectbox('🎮 GPU', ['Intel', 'AMD', 'Nvidia'])
#     os = st.selectbox('🖥️ OS', ['Windows', 'macOS', 'Linux', 'Chrome OS'])
#     weight = st.number_input('⚖️ Weight (kg)', 0.5, 5.0, 2.0, 0.1)


# # ─── Main panel ─────────────────────────────────────────────────────────────

# left, right = st.columns([2, 1])

# with left:
#     st.subheader("📋 Selected Configuration")
#     specs = pd.DataFrame({
#         'Component': ['Brand', 'Type', 'Screen', 'Resolution', 'CPU',
#                       'RAM', 'HDD', 'SSD', 'GPU', 'OS', 'Weight'],
#         'Value': [company, laptop_type, f"{screen_size}\"",
#                   resolution + (' | Touchscreen' if touchscreen else '') + (' | IPS' if ips else ''),
#                   cpu, f"{ram} GB", f"{hdd} GB", f"{ssd} GB", gpu, os, f"{weight} kg"]
#     })
#     st.dataframe(specs, use_container_width=True, hide_index=True)

# with right:
#     st.subheader("🎯 Get Prediction")

#     if st.button('🚀 Predict Price'):
#         if model is None:
#             st.error("Model not loaded. Run `python main.py` first.")
#         else:
#             try:
#                 # Build feature row
#                 x_res, y_res = map(int, resolution.split('x'))
#                 ppi = calculate_ppi(x_res, y_res, screen_size)
#                 ts = 1 if touchscreen else 0
#                 ip = 1 if ips else 0
#                 cpu_brand = extract_cpu_brand(cpu)
#                 gpu_brand = extract_gpu_brand(gpu)
#                 hdd_val, ssd_val = hdd, ssd

#                 # Label encode using same mapping used during training
#                 company_map = {'Acer': 0, 'Apple': 1, 'Asus': 2, 'Dell': 3, 'HP': 4,
#                                'Lenovo': 5, 'MSI': 6, 'Razer': 7, 'Samsung': 8, 'Toshiba': 9}
#                 type_map = {'2 in 1 Convertible': 0, 'Gaming': 1, 'Netbook': 2,
#                             'Notebook': 3, 'Ultrabook': 4, 'Workstation': 5}
#                 cpu_map = {'AMD': 0, 'Intel Core i3': 1, 'Intel Core i5': 2,
#                            'Intel Core i7': 3, 'Intel Other': 4, 'Other': 5}
#                 gpu_map = {'AMD': 0, 'Intel': 1, 'Nvidia': 2, 'Other': 3}
#                 os_map = {'Chrome OS': 0, 'Linux': 1, 'Windows': 2, 'macOS': 3}

#                 # query = np.array([[
#                 #     company_map.get(company, 0),
#                 #     type_map.get(laptop_type, 3),
#                 #     screen_size,
#                 #     ram,
#                 #     weight,
#                 #     ts, ip, ppi,
#                 #     cpu_map.get(cpu_brand, 2),
#                 #     hdd_val, ssd_val,
#                 #     gpu_map.get(gpu_brand, 1),
#                 #     os_map.get(os, 2)
#                 # ]])

#                 query = np.array([[
#     company_map.get(company, 0),
#     type_map.get(laptop_type, 3),
#     screen_size,        # Inches
#     ram,                # Ram
#     weight,             # Weight
#     ts,                 # Touchscreen
#     ip,                 # IPS
#     ppi,                # ppi
#     cpu_map.get(cpu_brand, 2),   # Cpu_brand
#     hdd_val,            # HDD
#     ssd_val,            # SSD
#     gpu_map.get(gpu_brand, 1),   # Gpu_brand
#     os_map.get(os, 2),  # OpSys
#     screen_size         # Inches again (as separate feature)
# ]])

#                 log_price = model.predict(query)[0]
#                 price = np.exp(log_price)

#                 st.markdown(f"""
#                 <div class='prediction-box'>
#                     <h3>Estimated Price</h3>
#                     <h1 style='font-size:52px; margin:10px 0'>₹{price:,.0f}</h1>
#                     <p>≈ ${price/83:,.0f} USD</p>
#                 </div>
#                 """, unsafe_allow_html=True)

#                 st.info(f"📊 Price Range: ₹{price*0.9:,.0f} — ₹{price*1.1:,.0f}")
#                 st.success("✅ Prediction complete!")

#                 # Session history
#                 if 'history' not in st.session_state:
#                     st.session_state.history = []
#                 st.session_state.history.append({
#                     'Config': f"{company} {laptop_type} {ram}GB",
#                     'Price (₹)': int(price)
#                 })

#             except Exception as e:
#                 st.error(f"❌ Error: {e}")


# # ─── Comparison Chart ────────────────────────────────────────────────────────

# if 'history' in st.session_state and len(st.session_state.history) > 1:
#     st.markdown("---")
#     st.subheader("📈 Comparison")
#     hist_df = pd.DataFrame(st.session_state.history)
#     st.bar_chart(hist_df.set_index('Config'))
#     if st.button("🗑️ Clear History"):
#         st.session_state.history = []
#         st.rerun()


# # ─── Footer Stats ────────────────────────────────────────────────────────────

# st.markdown("---")
# c1, c2, c3, c4 = st.columns(4)
# stats = [
#     ("🎯 Best R²", "86%", "Random Forest"),
#     ("📊 Dataset", "1,330", "Laptops"),
#     ("🤖 Models", "10", "Trained & Compared"),
#     ("⚙️ Features", "13", "Engineered"),
# ]
# for col, (title, val, sub) in zip([c1, c2, c3, c4], stats):
#     col.markdown(f"""
#     <div class='stat-card'>
#         <p style='color:#888; font-size:14px'>{title}</p>
#         <p style='font-size:28px; font-weight:bold; color:#667eea'>{val}</p>
#         <p style='color:#aaa; font-size:13px'>{sub}</p>
#     </div>
#     """, unsafe_allow_html=True)


import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
import os

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.prediction-box {
    background: linear-gradient(135deg, #667eea, #764ba2);
    padding: 30px; border-radius: 15px;
    color: white; text-align: center;
    box-shadow: 0 8px 24px rgba(0,0,0,0.2);
}
.stat-card {
    background: white; padding: 20px;
    border-radius: 10px; text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; font-size: 18px;
    border-radius: 10px; border: none;
    padding: 12px 30px; width: 100%;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_ram(x):    return int(re.search(r'(\d+)', str(x)).group(1))
def clean_weight(x): return float(re.search(r'([\d.]+)', str(x)).group(1))

def extract_cpu(x):
    for b in ['Intel Core i7', 'Intel Core i5', 'Intel Core i3']:
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
        if 'SSD' in str(x) or 'Flash' in str(x): ssd = int(m.group(1)) * 1024
        else: hdd = int(m.group(1)) * 1024
    return hdd, ssd

def extract_res(x):
    m = re.search(r'(\d{3,4})x(\d{3,4})', str(x))
    return (int(m.group(1)), int(m.group(2))) if m else (1920, 1080)


# ── Load model & encoders ─────────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    base = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    model_path   = os.path.join(base, 'models', 'best_model.pkl')
    encoder_path = os.path.join(base, 'models', 'label_encoders.pkl')
    cols_path    = os.path.join(base, 'models', 'feature_cols.pkl')

    if not os.path.exists(model_path):
        st.error(f"❌ Model not found. Run `uv run python main.py` first.")
        st.stop()

    model    = joblib.load(model_path)
    le_map   = joblib.load(encoder_path)
    feat_cols= joblib.load(cols_path)
    return model, le_map, feat_cols


model, le_map, feat_cols = load_artifacts()


# ── Sidebar inputs ────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🔧 Configure Laptop")

    company      = st.selectbox('🏢 Brand',
                    sorted(le_map['Company'].classes_))
    laptop_type  = st.selectbox('📱 Type',
                    sorted(le_map['TypeName'].classes_))
    screen_size  = st.slider('📏 Screen Size (inches)', 10.0, 18.0, 15.6, 0.1)
    resolution   = st.selectbox('🖥️ Resolution', [
                    '1920x1080','1366x768','2560x1440',
                    '3840x2160','2560x1600','3200x1800',
                    '2880x1800','2304x1440'])

    col1, col2   = st.columns(2)
    with col1:   touchscreen = st.checkbox('✋ Touchscreen')
    with col2:   ips         = st.checkbox('🎨 IPS')

    cpu   = st.selectbox('⚙️ CPU',
                sorted(le_map['Cpu_brand'].classes_))
    ram   = st.select_slider('🧠 RAM (GB)',
                [2,4,6,8,12,16,24,32,64], value=8)

    st.subheader("💾 Storage")
    c1, c2 = st.columns(2)
    with c1: hdd = st.selectbox('HDD (GB)', [0,128,256,512,1024,2048])
    with c2: ssd = st.selectbox('SSD (GB)', [0,8,128,256,512,1024], index=3)

    gpu    = st.selectbox('🎮 GPU',
                sorted(le_map['Gpu_brand'].classes_))
    os_sel = st.selectbox('🖥️ OS',
                sorted(le_map['OpSys'].classes_))
    weight = st.number_input('⚖️ Weight (kg)', 0.5, 5.0, 2.0, 0.1)


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("<h1 style='text-align:center;color:#667eea'>💻 Laptop Price Predictor</h1>",
            unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#888'>ML-powered price estimation</p>",
            unsafe_allow_html=True)
st.markdown("---")

left, right = st.columns([2, 1])

with left:
    st.subheader("📋 Selected Configuration")
    specs = pd.DataFrame({
        'Component': ['Brand','Type','Screen','Resolution','CPU',
                      'RAM','HDD','SSD','GPU','OS','Weight'],
        'Value':     [company, laptop_type, f"{screen_size}\"",
                      resolution +
                      (' | Touchscreen' if touchscreen else '') +
                      (' | IPS' if ips else ''),
                      cpu, f"{ram} GB", f"{hdd} GB",
                      f"{ssd} GB", gpu, os_sel, f"{weight} kg"]
    })
    st.dataframe(specs, use_container_width=True, hide_index=True)

with right:
    st.subheader("🎯 Get Prediction")

    if st.button('🚀 Predict Price'):
        try:
            # ── Build features exactly as trained ────────────────────────────
            x_res, y_res = extract_res(resolution)
            ppi          = ((x_res**2 + y_res**2)**0.5) / screen_size
            ts           = 1 if touchscreen else 0
            ip           = 1 if ips else 0
            cpu_brand    = extract_cpu(cpu)
            gpu_brand    = extract_gpu(gpu)
            hdd_v, ssd_v = hdd, ssd

            # Encode categoricals using SAVED encoders (matches training exactly)
            def safe_encode(le, val):
                classes = list(le.classes_)
                return classes.index(val) if val in classes else 0

            row = {
                'Company':     safe_encode(le_map['Company'],   company),
                'TypeName':    safe_encode(le_map['TypeName'],  laptop_type),
                'Inches':      screen_size,
                'Ram':         ram,
                'Weight':      weight,
                'Touchscreen': ts,
                'IPS':         ip,
                'ppi':         ppi,
                'Cpu_brand':   safe_encode(le_map['Cpu_brand'], cpu_brand),
                'HDD':         hdd_v,
                'SSD':         ssd_v,
                'Gpu_brand':   safe_encode(le_map['Gpu_brand'], gpu_brand),
                'OpSys':       safe_encode(le_map['OpSys'],     os_sel),
            }

            # Order columns exactly as trained
            query = pd.DataFrame([row])[feat_cols]

            # Predict
            log_price = model.predict(query)[0]
            price     = np.exp(log_price)

            st.markdown(f"""
            <div class='prediction-box'>
                <h3>Estimated Price</h3>
                <h1 style='font-size:52px;margin:10px 0'>₹{price:,.0f}</h1>
                <p>≈ ${price/83:,.0f} USD</p>
            </div>
            """, unsafe_allow_html=True)

            st.info(f"📊 Range: ₹{price*0.9:,.0f} — ₹{price*1.1:,.0f}")
            st.success("✅ Prediction complete!")

            if 'history' not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append({
                'Config':    f"{company} {laptop_type} {ram}GB {cpu}",
                'Price (₹)': int(price)
            })

        except Exception as e:
            st.error(f"❌ Error: {e}")


# ── Comparison chart ──────────────────────────────────────────────────────────

if 'history' in st.session_state and len(st.session_state.history) > 1:
    st.markdown("---")
    st.subheader("📈 Price Comparison")
    hist_df = pd.DataFrame(st.session_state.history)
    st.bar_chart(hist_df.set_index('Config'))
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()


# ── Footer stats ──────────────────────────────────────────────────────────────

st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
for col, (title, val, sub) in zip(
    [c1,c2,c3,c4],
    [("🎯 Best R²","86%","Random Forest"),
     ("📊 Dataset","1,330","Laptops"),
     ("🤖 Models","10","Trained"),
     ("⚙️ Features","13","Engineered")]
):
    col.markdown(f"""
    <div class='stat-card'>
        <p style='color:#888;font-size:14px'>{title}</p>
        <p style='font-size:28px;font-weight:bold;color:#667eea'>{val}</p>
        <p style='color:#aaa;font-size:13px'>{sub}</p>
    </div>
    """, unsafe_allow_html=True)