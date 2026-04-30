import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc
)

st.set_page_config(
    page_title="Tokopedia ML Pipeline",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# 1. GLOBAL STYLE
# ─────────────────────────────────────────────
def set_ui_style():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --primary-50:  #f0fdf4;
        --primary-100: #dcfce7;
        --primary-500: #22c55e;
        --primary-600: #16a34a;
        --primary-700: #15803d;
        --gray-50:   #f9fafb;
        --gray-100:  #f3f4f6;
        --gray-200:  #e5e7eb;
        --gray-400:  #9ca3af;
        --gray-600:  #4b5563;
        --gray-900:  #111827;
        --white:     #ffffff;
        --shadow-sm: 0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
        --shadow-md: 0 4px 16px rgba(0,0,0,.07), 0 2px 4px rgba(0,0,0,.04);
        --radius-sm: 6px;
        --radius-md: 10px;
        --radius-lg: 14px;
    }

    * { font-family: 'Inter', sans-serif !important; }

    .stApp { background: #f8fafc; }
    #MainMenu, footer { visibility: hidden; }
    
    .block-container {
        background-color: white;
        padding: 2.5rem 3.5rem !important;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.04);
        margin-top: 2rem;
        max-width: 1100px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--white) !important;
        border-right: 1px solid var(--gray-200) !important;
    }
    
    div[role="radiogroup"] { gap: .2rem; display: flex; flex-direction: column; }
    div[role="radiogroup"] > label > div:first-child { display: none !important; }
    div[role="radiogroup"] > label {
        padding: 10px 14px; border-radius: var(--radius-sm);
        transition: all .15s ease; cursor: pointer; width: 100%;
        border: 1px solid transparent;
    }
    div[role="radiogroup"] > label:hover { background: var(--gray-50); }
    div[role="radiogroup"] > label:has(input:checked) {
        background: var(--primary-50) !important;
        border-left: 3px solid var(--primary-600) !important;
    }
    div[role="radiogroup"] > label p { font-size: 14px; font-weight: 500; color: var(--gray-600); margin: 0; }
    div[role="radiogroup"] > label:has(input:checked) p { font-weight: 600 !important; color: var(--primary-700) !important; }

    [data-testid="stSidebar"] button {
        background: transparent !important; border: 1px solid var(--gray-400) !important;
        color: var(--gray-600) !important; border-radius: var(--radius-sm) !important;
        font-weight: 500 !important; font-size: 13px !important; transition: all .2s;
    }
    [data-testid="stSidebar"] button:hover { background: var(--gray-100) !important; color: var(--gray-900) !important; }
    
    button[kind="primary"] {
        background: var(--primary-600) !important; color: white !important; border: none !important;
        border-radius: var(--radius-sm) !important; font-weight: 500 !important; transition: all .2s !important;
    }
    button[kind="primary"]:hover { background: var(--primary-700) !important; }

    /* Clean UI Overrides */
    .card {
        background: transparent;
        border-radius: 0;
        box-shadow: none;
        padding: 0;
        margin-bottom: 2.5rem;
    }
    
    .section-title {
        font-size: 1.1rem; font-weight: 700; color: var(--gray-900);
        margin-bottom: 1.2rem; display: flex; align-items: center; gap: .5rem;
    }
    .section-title::after { content: ''; flex: 1; height: 1px; background: var(--gray-200); }

    /* Progress Step Bar */
    .step-bar { display: flex; gap: .5rem; margin-bottom: 2rem; }
    .step-item { flex: 1; padding: 10px; border-radius: var(--radius-sm); text-align: center; font-size: .75rem; font-weight: 600; letter-spacing: .02em; text-transform: uppercase; }
    .step-done    { background: var(--primary-50); color: var(--primary-700); border: 1px solid var(--primary-100); }
    .step-active  { background: var(--primary-600); color: white; }
    .step-pending { background: var(--gray-50);  color: var(--gray-400); border: 1px solid var(--gray-100); }

    /* Info Boxes for Explanations */
    .info-box { background: #f8fafc; border-left: 4px solid var(--primary-500); padding: 1rem 1.2rem; font-size: 0.95rem; color: var(--gray-600); margin-top: 1rem; border-radius: 0 8px 8px 0; line-height: 1.5; }
    .warn-box { background: #fffbeb; border-left: 4px solid #f59e0b; padding: 1rem 1.2rem; font-size: 0.95rem; color: #92400e; margin-bottom: 1.5rem; }
    
    .pred-box { background: var(--gray-50); border: 1px solid var(--gray-200); border-radius: var(--radius-md); padding: 2.5rem; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# STEP PROGRESS BAR & DATA LOADING
# ─────────────────────────────────────────────
STEPS = ["EDA", "Preprocessing", "Model", "Evaluation", "Testing"]

def step_status(step_name):
    checks = {
        "EDA":          st.session_state.get('feature_confirmed', False),
        "Preprocessing":st.session_state.get('preprocessing_done', False),
        "Model":        st.session_state.get('model_trained', False),
        "Evaluation":   st.session_state.get('model_trained', False),
        "Testing":      st.session_state.get('model_trained', False),
    }
    return checks.get(step_name, False)

def render_progress(active):
    html = '<div class="step-bar">'
    for i, s in enumerate(STEPS, 1):
        done   = step_status(s)
        is_act = (s == active)
        cls    = "step-active" if is_act else ("step-done" if done else "step-pending")
        html  += f'<div class="step-item {cls}">{i}. {s}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("produk_tokopedia.csv")

    def prep_terjual(x):
        if pd.isna(x): return 0
        t = str(x).lower().replace('terjual','').replace('+','').replace(' ','')
        if 'rb' in t:
            try: return int(float(t.replace('rb','')) * 1000)
            except: return 0
        elif 'jt' in t:
            try: return int(float(t.replace('jt','')) * 1_000_000)
            except: return 0
        else:
            try: return int(float(t))
            except: return 0

    def prep_ulasan(x):
        if pd.isna(x): return 0
        t = str(x).lower().replace('ulasan','').replace('+','').replace(' ','')
        try: return int(float(t))
        except: return 0

    df['Terjual_bersih']     = df['Terjual'].apply(prep_terjual)
    df['Ulasan_bersih']      = df['Jumlah Ulasan'].apply(prep_ulasan)
    df['Harga_setelah_diskon']= df['Harga (IDR)'] * (1 - df['Diskon (%)']/100)
    df['Ada_diskon']          = (df['Diskon (%)'] > 0).astype(int)
    df['Skor_kepercayaan']    = df['Rating'] * df['Ulasan_bersih']

    Q1 = df['Harga (IDR)'].quantile(0.01)
    Q3 = df['Harga (IDR)'].quantile(0.99)
    df = df[(df['Harga (IDR)'] >= Q1) & (df['Harga (IDR)'] <= Q3)]

    THRESHOLD = df['Terjual_bersih'].quantile(0.75)
    df['is_bestseller'] = (df['Terjual_bersih'] > THRESHOLD).astype(int)

    return df, THRESHOLD

# ─────────────────────────────────────────────
# 2. HOME
# ─────────────────────────────────────────────
def show_home():
    
    # --- HEADER / HERO SECTION ---
    st.markdown("""
    <div style="background-color: #ffffff; padding: 2.5rem 0; border-bottom: 4px solid #16a34a; margin-bottom: 2.5rem;">
        <h1 style="margin: 0; font-size: 2.4rem; font-weight: 800; color: #111827; letter-spacing: -0.02em;">Machine Learning Pipeline</h1>
        <p style="margin: 0.5rem 0 0; font-size: 1.15rem; font-weight: 400; color: #4b5563;">Prediksi Penjualan Produk E-Commerce (Studi Kasus: Tokopedia)</p>
    </div>
    """, unsafe_allow_html=True)

    # --- SECTION 1: LATAR BELAKANG ---
    st.markdown("""
    <div style="padding: 2rem; background: #fafafa; border-radius: 12px; border-left: 6px solid #16a34a; margin-bottom: 2rem;">
        <h2 style="margin-top: 0; font-size: 1.3rem; font-weight: 700; color: #111827; margin-bottom: 1rem;">Latar Belakang & Tujuan Proyek</h2>
        <p style="color: #4b5563; line-height: 1.6; margin-bottom: 1rem;">Aplikasi ini merupakan simulasi dari <b>End-to-End Machine Learning Pipeline</b> untuk menganalisis dan memprediksi performa penjualan produk. Sistem ini memfasilitasi ekstraksi fitur dari teks kotor, pemrosesan awal (preprocessing), hingga pelatihan algoritma secara interaktif (AutoML).</p>
        <p style="color: #4b5563; line-height: 1.6; margin: 0;">Tujuan utama sistem ini adalah membantu pengambilan keputusan bisnis dengan mengklasifikasikan apakah suatu produk memiliki potensi untuk menjadi <b>Best Seller</b> berdasarkan spesifikasi harga, rating, ulasan, dan metrik lainnya.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- SECTION 2: DEFINISI TARGET ---
    st.markdown("""
    <div style="padding: 2rem; background: #fafafa; border-radius: 12px; border-left: 6px solid #22c55e; margin-bottom: 2rem;">
        <h2 style="margin-top: 0; font-size: 1.3rem; font-weight: 700; color: #111827; margin-bottom: 1rem;">Definisi Variabel Target</h2>
        <p style="color: #4b5563; line-height: 1.6; margin-bottom: 1.2rem;">Sistem ini menggunakan pendekatan <b>Klasifikasi Biner (Binary Classification)</b>:</p>
        <div style="display: flex; flex-direction: column; gap: 1rem;">
            <div style="background: #ffffff; padding: 1.2rem; border-radius: 8px; border: 1px solid #dcfce7;">
                <strong style="color: #15803d; font-size: 1.1rem;">1 (Best Seller)</strong><br>
                <span style="color: #4b5563; font-size: 0.95rem;">Volume penjualan produk berada pada persentil atas (Top 25% pasar). Menandakan produk yang sangat diminati konsumen.</span>
            </div>
            <div style="background: #ffffff; padding: 1.2rem; border-radius: 8px; border: 1px solid #f3f4f6;">
                <strong style="color: #374151; font-size: 1.1rem;">0 (Reguler)</strong><br>
                <span style="color: #4b5563; font-size: 0.95rem;">Volume penjualan reguler atau berada di bawah ambang batas (threshold) rata-rata pasar.</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- SECTION 3: ALUR SISTEM ---
    st.markdown("""
    <div style="padding: 2rem; background: #fafafa; border-radius: 12px; border-left: 6px solid #4ade80; margin-bottom: 2rem;">
        <h2 style="margin-top: 0; font-size: 1.3rem; font-weight: 700; color: #111827; margin-bottom: 1.2rem;">Alur Sistem (Workflow)</h2>
        <div style="display: flex; flex-direction: column; gap: 0.8rem; color: #4b5563;">
            <div><b style="color: #111827;">1. EDA (Exploratory Data Analysis)</b> — Eksplorasi data dan analisis awal</div>
            <div><b style="color: #111827;">2. Preprocessing</b> — Cleaning, splitting, dan scaling data</div>
            <div><b style="color: #111827;">3. Model Training</b> — Melatih model Machine Learning</div>
            <div><b style="color: #111827;">4. Evaluation</b> — Mengukur performa model</div>
            <div><b style="color: #111827;">5. Testing</b> — Prediksi data simulasi bisnis</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 3. EDA & FEATURE SELECTION
# ─────────────────────────────────────────────
def show_eda():
    st.title("Exploratory Data Analysis")
    render_progress("EDA")

    df, thresh = load_and_clean_data()
    num_cols = ['Harga (IDR)', 'Diskon (%)', 'Rating', 'Ulasan_bersih', 'Harga_setelah_diskon', 'Skor_kepercayaan']

    # Dataset Overview
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Tinjauan Dataset</div>', unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Data Produk", f"{len(df):,}")
    m2.metric("Produk Best Seller", f"{df['is_bestseller'].sum():,}")
    m3.metric("Threshold Penjualan", f"{thresh:,.0f} unit")
    m4.metric("Rasio Best Seller", f"{(df['is_bestseller'].mean() * 100):.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    chk1, chk2, chk3 = st.columns(3)
    show_head  = chk1.checkbox("Tampilkan Sampel Data")
    show_info  = chk2.checkbox("Tampilkan Struktur Data (df.info)")
    show_desc  = chk3.checkbox("Tampilkan Statistik Deskriptif")

    if show_head: st.dataframe(df[['Nama Produk','Harga (IDR)','Diskon (%)','Rating','Terjual_bersih','is_bestseller']].head(), use_container_width=True)
    if show_info:
        buf = io.StringIO(); df.info(buf=buf); st.code(buf.getvalue(), language="text")
    if show_desc: st.dataframe(df[num_cols].describe().style.format("{:.2f}"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Univariate
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Analisis Univariat</div>', unsafe_allow_html=True)
    col_u1, col_u2 = st.columns(2)
    uni_feat = col_u1.selectbox("Pilih Variabel:", num_cols, key="uni_feat")
    uni_plot = col_u2.selectbox("Jenis Visualisasi:", ["Histogram", "Boxplot"], key="uni_plot")

    if uni_plot == "Histogram":
        fig_u = px.histogram(df, x=uni_feat, color='is_bestseller', barmode='overlay', opacity=0.7,
                             color_discrete_map={0:"#94a3b8", 1:"#16a34a"}, labels={'is_bestseller':'Kelas Target'})
        st.plotly_chart(fig_u, use_container_width=True)
        st.markdown('<div class="info-box"><b>Interpretasi Histogram:</b> Grafik ini menunjukkan sebaran frekuensi data. Distribusi yang menumpuk di satu sisi menunjukkan adanya *skewness* (kemiringan), yang mengindikasikan bahwa sebagian besar produk berada pada rentang nilai tersebut.</div>', unsafe_allow_html=True)
    else:
        fig_u = px.box(df, x=df['is_bestseller'].astype(str), y=uni_feat, color=df['is_bestseller'].astype(str),
                       color_discrete_map={"0":"#94a3b8","1":"#16a34a"}, labels={'x':'Kelas Target'})
        st.plotly_chart(fig_u, use_container_width=True)
        st.markdown('<div class="info-box"><b>Interpretasi Boxplot:</b> Grafik ini berguna untuk melihat nilai median serta mengidentifikasi keberadaan *outlier*. Perbedaan posisi kotak antar kelas menunjukkan bahwa fitur ini mungkin berpengaruh kuat terhadap target klasifikasi.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Bivariate
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Analisis Bivariat</div>', unsafe_allow_html=True)
    col_b1, col_b2 = st.columns(2)
    sumbu_x = col_b1.selectbox("Variabel Sumbu X:", num_cols, index=0)
    sumbu_y = col_b2.selectbox("Variabel Sumbu Y:", ['Terjual_bersih'] + num_cols, index=0)
    
    fig_b = px.scatter(df, x=sumbu_x, y=sumbu_y, color=df['is_bestseller'].astype(str),
                       color_discrete_map={"0":"#cbd5e1","1":"#16a34a"}, opacity=0.7)
    st.plotly_chart(fig_b, use_container_width=True)
    st.markdown('<div class="info-box"><b>Interpretasi Scatter Plot:</b> Membantu memvisualisasikan korelasi atau pola hubungan antara dua variabel. Jika titik-titik hijau (Best Seller) membentuk klaster di area tertentu, ini menunjukkan adanya pola kuat yang dapat dipelajari oleh model.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Heatmap
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Matriks Korelasi (Heatmap)</div>', unsafe_allow_html=True)
    corr = df[num_cols + ['is_bestseller']].corr()
    fig_h = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale='Greens')
    st.plotly_chart(fig_h, use_container_width=True)
    st.markdown('<div class="info-box"><b>Interpretasi Heatmap Korelasi:</b> Angka yang mendekati 1.00 atau -1.00 menandakan korelasi linear yang kuat. Variabel yang memiliki korelasi tinggi terhadap "is_bestseller" adalah fitur yang paling krusial.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Feature Selection
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Seleksi Fitur (Feature Selection)</div>', unsafe_allow_html=True)
    ALL_FEATS = ['Harga (IDR)', 'Diskon (%)', 'Rating', 'Ulasan_bersih', 'Harga_setelah_diskon', 'Ada_diskon', 'Skor_kepercayaan']
    
    use_rec = st.checkbox("Gunakan seluruh fitur secara default", value=True)
    default_sel = ALL_FEATS if use_rec else st.session_state.get('selected_features', [])
    sel_feats = st.multiselect("Variabel Independen (X):", ALL_FEATS, default=default_sel)

    if st.button("Simpan Konfigurasi Fitur", type="primary"):
        if sel_feats:
            st.session_state['selected_features'] = sel_feats
            st.session_state['feature_confirmed'] = True
            st.success("Konfigurasi fitur berhasil disimpan. Silakan lanjutkan ke tahap Preprocessing.")
        else:
            st.error("Sistem membutuhkan minimal satu variabel untuk melanjutkan.")
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 4. PREPROCESSING
# ─────────────────────────────────────────────
def show_preprocessing():
    st.title("Data Preprocessing")
    render_progress("Preprocessing")

    if not st.session_state.get('feature_confirmed'):
        st.markdown('<div class="warn-box">Selesaikan tahap Feature Selection pada menu EDA terlebih dahulu.</div>', unsafe_allow_html=True)
        return

    df, _ = load_and_clean_data()
    X = df[st.session_state['selected_features']]
    y = df['is_bestseller']

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Parameter Pemisahan & Transformasi Data</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    test_size = col1.slider("Proporsi Data Uji (Test Size %)", 10, 40, 20) / 100
    random_state = col2.number_input("Random State", value=42)
    scaler_type = col3.selectbox("Metode Normalisasi/Standardisasi:", ["StandardScaler", "MinMaxScaler", "RobustScaler"])

    if st.button("Eksekusi Splitting & Scaling", type="primary"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        
        if scaler_type == "StandardScaler": scaler = StandardScaler()
        elif scaler_type == "MinMaxScaler": scaler = MinMaxScaler()
        else: scaler = RobustScaler()
        
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        
        st.session_state['preprocessed_data'] = {
            'X_train': X_train_scaled, 'X_test': X_test_scaled,
            'y_train': y_train.reset_index(drop=True), 'y_test': y_test.reset_index(drop=True)
        }
        st.session_state['scaler'] = scaler
        st.session_state['preprocessing_done'] = True 
        st.success("Pemrosesan data telah selesai.")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.get('preprocessing_done'):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Inspeksi Hasil Data Transformasi</div>', unsafe_allow_html=True)
        data = st.session_state['preprocessed_data']
        t1, t2, t3, t4 = st.tabs(["X_train (Pelatihan)", "X_test (Pengujian)", "y_train", "y_test"])
        with t1: st.write(f"Dimensi: {data['X_train'].shape}"); st.dataframe(data['X_train'].head())
        with t2: st.write(f"Dimensi: {data['X_test'].shape}"); st.dataframe(data['X_test'].head())
        with t3: st.write(f"Dimensi: {data['y_train'].shape}"); st.dataframe(data['y_train'].head())
        with t4: st.write(f"Dimensi: {data['y_test'].shape}"); st.dataframe(data['y_test'].head())
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 5. MODEL TRAINING
# ─────────────────────────────────────────────
def show_model():
    st.title("Model Training")
    render_progress("Model")

    if not st.session_state.get('preprocessing_done'):
        st.markdown('<div class="warn-box">Tahap pemrosesan data (Preprocessing) belum diselesaikan.</div>', unsafe_allow_html=True)
        return

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Konfigurasi Algoritma</div>', unsafe_allow_html=True)
    model_choice = st.selectbox("Pilihan Algoritma:", ["Random Forest", "XGBoost", "Logistic Regression"])
    
    col1, col2 = st.columns(2)
    if model_choice == "Random Forest":
        n_est = col1.select_slider("n_estimators", options=[50, 100, 150, 200, 300], value=100)
        m_depth = col2.slider("max_depth", 2, 20, 10)
        model = RandomForestClassifier(n_estimators=n_est, max_depth=m_depth, random_state=42, class_weight='balanced')
    elif model_choice == "XGBoost":
        n_est = col1.select_slider("n_estimators", options=[50, 100, 150, 200], value=100)
        lr = col1.selectbox("learning_rate", [0.01, 0.05, 0.1, 0.2])
        m_depth = col2.slider("max_depth", 2, 10, 5)
        model = XGBClassifier(n_estimators=n_est, learning_rate=lr, max_depth=m_depth, random_state=42, eval_metric='logloss')
    else:
        c_val = col1.selectbox("Regularization (C)", [0.01, 0.1, 1.0, 10.0])
        m_iter = col2.select_slider("max_iter", options=[100, 200, 500], value=100)
        model = LogisticRegression(C=c_val, max_iter=m_iter, random_state=42, class_weight='balanced')

    if st.button("Mulai Pelatihan Model", type="primary"):
        with st.status("Menjalankan proses pelatihan...", expanded=True) as status:
            st.write("Mengalokasikan data pelatihan...")
            X_train = st.session_state['preprocessed_data']['X_train']
            y_train = st.session_state['preprocessed_data']['y_train']
            time.sleep(0.5)
            
            st.write(f"Melatih algoritma {model_choice}...")
            model.fit(X_train, y_train)
            time.sleep(0.5)
            
            st.session_state['trained_model'] = model
            st.session_state['model_name'] = model_choice
            st.session_state['model_trained'] = True
            status.update(label="Pelatihan selesai.", state="complete", expanded=False)
        
        st.success(f"Model {model_choice} telah berhasil dilatih dan disimpan ke dalam sesi memori.")
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────
def show_evaluation():
    st.title("Model Evaluation")
    render_progress("Evaluation")

    if not st.session_state.get('model_trained'):
        st.markdown('<div class="warn-box">Model belum dilatih. Silakan kembali ke menu Model Training.</div>', unsafe_allow_html=True)
        return

    model = st.session_state['trained_model']
    X_test = st.session_state['preprocessed_data']['X_test']
    y_test = st.session_state['preprocessed_data']['y_test']
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    eval_options = st.multiselect(
        "Pilih Metrik & Visualisasi Evaluasi:", 
        ["Classification Report", "Confusion Matrix", "ROC-AUC Curve", "Feature Importance"],
        default=["Classification Report", "Confusion Matrix"]
    )

    if "Classification Report" in eval_options:
        st.markdown('<div class="section-title">Laporan Klasifikasi (Classification Report)</div>', unsafe_allow_html=True)
        report = classification_report(y_test, y_pred, target_names=["Reguler", "Best Seller"], output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().style.format(precision=3), use_container_width=True)
        st.markdown('<div class="info-box"><b>Interpretasi Metrik:</b> <b>Precision</b> mengukur akurasi prediksi positif, <b>Recall</b> mengukur kemampuan model menemukan semua data positif, dan <b>F1-Score</b> adalah keseimbangan antara keduanya. Gunakan akurasi keseluruhan secara hati-hati jika proporsi data tidak seimbang.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    if "Confusion Matrix" in eval_options:
        with col1:
            st.markdown('<br><div class="section-title">Matriks Kebingungan (Confusion Matrix)</div>', unsafe_allow_html=True)
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Greens', x=["Reguler", "Best Seller"], y=["Reguler", "Best Seller"])
            st.plotly_chart(fig_cm, use_container_width=True)
            st.markdown('<div class="info-box"><b>Interpretasi:</b> Diagonal utama adalah prediksi yang tepat (True Positives & Negatives). Kotak di luar diagonal menunjukkan jumlah prediksi yang meleset.</div>', unsafe_allow_html=True)

    if "ROC-AUC Curve" in eval_options:
        with col2:
            st.markdown('<br><div class="section-title">Kurva ROC-AUC</div>', unsafe_allow_html=True)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig_roc = go.Figure(go.Scatter(x=fpr, y=tpr, mode='lines', fill='tozeroy', name=f"AUC = {auc(fpr, tpr):.3f}", line=dict(color="#16a34a")))
            fig_roc.add_shape(type='line', x0=0, x1=1, y0=0, y1=1, line=dict(color="gray", dash="dash"))
            st.plotly_chart(fig_roc, use_container_width=True)
            st.markdown('<div class="info-box"><b>Interpretasi:</b> Semakin kurva hijau melengkung ke arah sudut kiri atas (AUC mendekati 1.00), semakin baik model dalam membedakan antara kedua kelas.</div>', unsafe_allow_html=True)

    if "Feature Importance" in eval_options and st.session_state['model_name'] in ["Random Forest", "XGBoost"]:
        st.markdown('<br><div class="section-title">Signifikansi Variabel (Feature Importance)</div>', unsafe_allow_html=True)
        importances = model.feature_importances_
        df_imp = pd.DataFrame({"Fitur": X_test.columns, "Kepentingan": importances}).sort_values(by="Kepentingan", ascending=True)
        fig_imp = px.bar(df_imp, x="Kepentingan", y="Fitur", orientation='h', color_discrete_sequence=['#16a34a'])
        st.plotly_chart(fig_imp, use_container_width=True)
        st.markdown('<div class="info-box"><b>Interpretasi:</b> Menampilkan kontribusi tiap variabel. Semakin panjang batang suatu fitur, semakin besar bobotnya dalam keputusan klasifikasi yang dilakukan oleh algoritma.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 7. TESTING (SIMULASI) MENGGUNAKAN LOGIKA FEATURE ENGINEERING
# ─────────────────────────────────────────────
def show_testing():
    st.title("Testing & Simulation")
    render_progress("Testing")

    if not st.session_state.get('model_trained'):
        st.markdown('<div class="warn-box">Model belum dilatih. Sistem tidak dapat melakukan prediksi.</div>', unsafe_allow_html=True)
        return

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Input Spesifikasi Produk Dasar</div>', unsafe_allow_html=True)
    st.write("Sistem akan secara otomatis melakukan kalkulasi *Feature Engineering* (seperti Skor Kepercayaan dan Harga Setelah Diskon) berdasarkan 4 input di bawah ini.")
    
    col1, col2 = st.columns(2)
    with col1:
        input_harga = st.number_input("Harga Dasar (IDR)", min_value=1000, value=150000, step=5000)
        input_diskon = st.slider("Persentase Diskon (%)", 0, 100, 10)
    with col2:
        input_rating = st.slider("Rating Produk", 0.0, 5.0, 4.8, 0.1)
        input_ulasan = st.number_input("Jumlah Ulasan Pembeli", min_value=0, value=150, step=10)

    if st.button("Lakukan Prediksi AI", type="primary", use_container_width=True):
        
        harga_diskon = input_harga * (1 - input_diskon / 100)
        ada_diskon = 1 if input_diskon > 0 else 0
        skor_kepercayaan = input_rating * input_ulasan
        
        all_inputs = {
            'Harga (IDR)': input_harga,
            'Diskon (%)': input_diskon,
            'Rating': input_rating,
            'Ulasan_bersih': input_ulasan,
            'Harga_setelah_diskon': harga_diskon,
            'Ada_diskon': ada_diskon,
            'Skor_kepercayaan': skor_kepercayaan
        }
        
        X_train_cols = st.session_state['preprocessed_data']['X_train'].columns
        final_input_dict = {col: all_inputs[col] for col in X_train_cols}
        
        input_df = pd.DataFrame([final_input_dict])
        input_scaled = st.session_state['scaler'].transform(input_df)
        
        pred = st.session_state['trained_model'].predict(input_scaled)[0]
        proba = st.session_state['trained_model'].predict_proba(input_scaled)[0]
        
        pred_label = "BEST SELLER" if pred == 1 else "REGULER"
        confidence = max(proba) * 100
        color = "#16a34a" if pred == 1 else "#4b5563"
        
        st.markdown("<hr style='margin: 2rem 0; border: 1px solid #e5e7eb;'>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="pred-box">
            <h4 style="color: #64748b; font-weight: 500; margin-bottom: 0;">Hasil Klasifikasi Model</h4>
            <h1 style="font-size: 3.5rem; color: {color}; margin: 10px 0;">{pred_label}</h1>
            <p style="color: #475569; font-size: 1.1rem; margin-top: 0;">Tingkat Keyakinan (Confidence Score): <b>{confidence:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN APP ROUTING
# ─────────────────────────────────────────────
def main():
    set_ui_style()
    if 'current_page' not in st.session_state: st.session_state['current_page'] = "Home"

    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    menu = ["Home", "EDA", "Preprocessing", "Model", "Evaluation", "Testing"]
    choice = st.sidebar.radio("Navigasi Utama", menu, key="current_page", label_visibility="collapsed")

    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    if st.sidebar.button("Reset Sistem", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    if choice == "Home": show_home()
    elif choice == "EDA": show_eda()
    elif choice == "Preprocessing": show_preprocessing()
    elif choice == "Model": show_model()
    elif choice == "Evaluation": show_evaluation()
    elif choice == "Testing": show_testing()

if __name__ == "__main__":
    main()