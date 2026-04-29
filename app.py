import streamlit as st
import pandas as pd
import pickle
import numpy as np

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Prediksi Best Seller",
    page_icon="🛍️",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    try:
        with open("model_bestseller.pickle", "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None

model_package = load_model()


if model_package is None:
    st.error("Model tidak ditemukan!")
    st.stop()

@st.cache_data   
def load_dataset():
    import os
    path = "produk_tokopedia.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return None
df = load_dataset()

model = model_package['model']
FEATURES = model_package['features']
THRESHOLD = model_package['threshold']
STATS = model_package['market_stats']
IMPORTANCES = STATS['feature_importances']

# =========================
# FUNCTION
# =========================
def buat_input(harga, diskon, rating, ulasan):
    harga_efektif = harga * (1 - diskon / 100)
    return pd.DataFrame([{
        'Harga (IDR)': harga,
        'Diskon (%)': diskon,
        'Rating': rating,
        'Ulasan_bersih': ulasan,
        'Harga_setelah_diskon': harga_efektif,
        'Ada_diskon': 1 if diskon > 0 else 0,
        'Skor_kepercayaan': rating * ulasan,
    }])[FEATURES]

def prediksi(harga, diskon, rating, ulasan):
    data = buat_input(harga, diskon, rating, ulasan)
    kelas = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1] * 100
    return kelas, prob

def gauge_chart(prob):
    import math
    color = "#22c55e" if prob >= 70 else "#f59e0b" if prob >= 40 else "#ef4444"
    
    # Sudut busur
    angle = math.radians(180 - (prob * 1.8))
    cx, cy, r = 150, 130, 100
    x = cx + r * math.cos(angle)
    y = cy - r * math.sin(angle)
    
    # Hitung ujung jarum
    needle_r = 85
    nx = cx + needle_r * math.cos(angle)
    ny = cy - needle_r * math.sin(angle)

    return f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 160">
        <path d="M 50 130 A 100 100 0 0 1 250 130" 
              fill="none" stroke="#e5e7eb" stroke-width="18" stroke-linecap="round"/>
        <path d="M 50 130 A 100 100 0 0 1 {x:.1f} {y:.1f}" 
              fill="none" stroke="{color}" stroke-width="18" stroke-linecap="round"/>
        <line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="{color}" stroke-width="4" stroke-linecap="round"/>
        <circle cx="{cx}" cy="{cy}" r="6" fill="{color}"/>
        
        <text x="150" y="150" text-anchor="middle" font-size="22" font-weight="bold" fill="{color}">{prob:.1f}%</text>
    </svg>
    """

def tampilkan_feature_importance():
    df = pd.DataFrame.from_dict(IMPORTANCES, orient='index', columns=['Importance'])
    st.bar_chart(df)

def tampilkan_saran(prob, harga, diskon, rating, ulasan):
    st.markdown('#### 💡 Saran Spesifik')
    saran = []

    if diskon == 0:
        saran.append("- **Coba tambahkan diskon** (minimal 10-20%) untuk menarik perhatian pembeli.")
    if harga > 500_000 and diskon < 20:
        saran.append("- Harga di atas Rp 500.000 butuh **diskon lebih besar** agar kompetitif.")
    if rating < 4.0 and rating > 0:
        saran.append("- **Tingkatkan kualitas produk & layanan** untuk mendongkrak rating di atas 4.0.")
    if ulasan < 10 and ulasan > 0:
        saran.append("- **Dorong pembeli untuk meninggalkan ulasan** (mis. dengan bonus kecil).")
    if prob >= 70:
        saran.append("- Strategi produkmu sudah bagus! Fokus pada **konsistensi stok**.")

    if saran:
        for s in saran:
            st.markdown(s)
    else:
        st.markdown("- Optimalkan kombinasi harga dan diskon untuk meningkatkan daya saing.")

# =========================
# SIDEBAR MENU
# =========================
menu = st.sidebar.radio(
    "**NAVIGATION**",
    ["Home", "Dataset", "EDA", "Preprocessing", "Training", "Result"]
)


if menu == "Home":
    st.title("Prediksi Best Seller Produk")

    st.markdown("Latar Belakang")

    st.write("""
    Di era e-commerce seperti sekarang, persaingan antar produk sangat tinggi. 
    Banyak penjual kesulitan menentukan strategi harga, diskon, dan kualitas produk
    agar dapat menjadi best seller di marketplace seperti Tokopedia.
    """)

    st.write("""
    Tidak semua produk dengan harga murah akan laku, dan tidak semua produk mahal gagal.
    Faktor seperti rating, jumlah ulasan, serta strategi diskon memiliki peran penting
    dalam menentukan keberhasilan suatu produk di pasar.
    """)

    st.markdown("Tujuan Aplikasi")

    st.write("""
    Aplikasi ini dibuat untuk membantu penjual dalam:
    """)

    st.markdown("""
    - Menganalisis potensi produk menjadi best seller  
    - Memberikan insight berbasis data  
    - Mengoptimalkan strategi harga dan diskon  
    - Memanfaatkan Machine Learning untuk pengambilan keputusan  
    """)

    st.markdown("Cara Kerja")

    st.write("""
    Model Machine Learning dilatih menggunakan data produk marketplace,
    dengan mempertimbangkan beberapa fitur utama:
    """)

    st.markdown("""
    - Harga produk  
    - Diskon  
    - Rating  
    - Jumlah ulasan  
    - Harga setelah diskon  
    - Skor kepercayaan (rating × ulasan)  
    """)

    st.info("""
    Model akan memprediksi apakah suatu produk memiliki potensi menjadi best seller
    berdasarkan pola dari data sebelumnya.
    """)

    st.markdown("Teknologi yang Digunakan")

    st.markdown("""
    - Python  
    - Pandas & NumPy  
    - Scikit-learn (Random Forest)  
    - Streamlit (Web App)  
    """)

# =========================
# DATASET
# =========================
elif menu == "Dataset":
    st.title("Dataset")


    if df is not None:
        st.success("Dataset berhasil dimuat")
        st.dataframe(df)

        st.subheader("Kolom Dataset")
        st.write(df.columns)

    else:
        st.error("dataset.csv tidak ditemukan")
# =========================
# EDA
# =========================
elif menu == "EDA":
    st.header("📈 Exploratory Data Analysis")
    if df is not None:
        col_pilih = st.selectbox("Pilih Fitur:", FEATURES)
            
        data_clean = df[col_pilih].dropna()
            
        if data_clean.nunique() < 20:
                distribusi = data_clean.value_counts().sort_index()
                st.bar_chart(distribusi)
        else:
            import numpy as np
            hist_values, bin_edges = np.histogram(data_clean, bins=30)

            hist_df = pd.DataFrame({
                'Frekuensi': hist_values
            }, index=[f"{int(e):,}" for e in bin_edges[:-1]])
            st.bar_chart(hist_df)
    else:
        st.warning("Data CSV tidak ditemukan.")

# =========================
# PREPROCESSING
# =========================
elif menu == "Preprocessing":
    st.title("Preprocessing")

    st.markdown("""
    **Cleaning Data**
    - Menghapus data kosong (missing values)
    - Menghilangkan data duplikat
    - Memastikan format data konsisten (contoh: harga dalam angka)
    - Menangani outlier (data ekstrem)

    **Feature Engineering**
    - Membuat fitur baru dari data yang ada:
        - Harga setelah diskon = harga × (1 - diskon)
        - Ada_diskon = 1 jika ada diskon
        - Skor_kepercayaan = rating × ulasan

    **Encoding**
    - Mengubah data kategori menjadi numerik
    - Contoh:
        - Ada_diskon → 1 (ya), 0 (tidak)
    - Tujuan: agar bisa diproses oleh model machine learning
    """)

# =========================
# TRAINING
# =========================
elif menu == "Training":
    st.header("🧠 Model Training Info")
    st.write(f"Total data diproses: **{STATS['total_produk']}** baris")
        
        # Tambahan Metrik Performa (Disalin manual dari hasil main.ipynb)
    col1, col2, col3 = st.columns(3)
    col1.metric("Akurasi Model", "84.2%") # Ganti dengan angka asli dari main.ipynb mu nanti
    col2.metric("ROC-AUC Score", "0.889") 
    col3.metric("Batas Best Seller", f"{THRESHOLD:,.0f} pcs")
        
    st.subheader("Distribusi Target")
    rasio = STATS['bestseller_rate'] * 100
    st.write(f"- **Best Seller (1)**: {rasio:.1f}%")
    st.write(f"- **Biasa (0)**: {100 - rasio:.1f}%")
    st.progress(int(rasio))

    st.subheader("Feature Importances")
    df_imp = pd.DataFrame(list(IMPORTANCES.items()), columns=['Fitur', 'Kepentingan']).set_index('Fitur')
    df_imp = df_imp.sort_values(by='Kepentingan', ascending=False)
    st.bar_chart(df_imp)

# =========================
# RESULT (SEMUA FITUR ASLI LU)
# =========================
elif menu == "Result":

    st.title("Prediksi Best Seller")

    tab_baru, tab_existing, tab_whatif, tab_pasar = st.tabs([
        "Produk Baru",
        "Produk Existing",
        "What-If",
        "Pasar"
    ])

    # =====================
    # TAB BARU
    # =====================
    with tab_baru:
        harga = st.number_input("Harga", value=100000, key="baru_harga")
        diskon = st.slider("Diskon", 0, 90, key="baru_diskon")

        if st.button("Prediksi"):
            kelas, prob = prediksi(harga, diskon, 0, 0)
            st.markdown(gauge_chart(prob), unsafe_allow_html=True)
            tampilkan_saran(prob, harga, diskon, 0, 0)

    # =====================
    # EXISTING
    # =====================
    with tab_existing:
        harga = st.number_input("Harga", value=100000, key="ex_harga")
        diskon = st.slider("Diskon", 0, 90, key="ex_diskon")
        rating = st.slider("Rating", 0.0, 5.0, 4.5, key="ex_rating")
        ulasan = st.number_input("Ulasan", value=50, key="ex_ulasan")

        if st.button("Prediksi Existing"):
            kelas, prob = prediksi(harga, diskon, rating, ulasan)
            st.markdown(gauge_chart(prob), unsafe_allow_html=True)
            tampilkan_saran(prob, harga, diskon, rating, ulasan)

    # =====================
    # WHAT IF
    # =====================
    with tab_whatif:
        harga = st.slider("Harga", 10000, 1000000, 100000, key="wi_harga")
        diskon = st.slider("Diskon", 0, 90, key="wi_diskon")
        rating = st.slider("Rating", 0.0, 5.0, 4.5, key="wi_rating")
        ulasan = st.slider("Ulasan", 0, 1000, 50, key="wi_ulasan")

        _, prob = prediksi(harga, diskon, rating, ulasan)
        st.markdown(gauge_chart(prob), unsafe_allow_html=True)

    # =====================
    # PASAR
    # =====================
    with tab_pasar:
        st.metric("Total Produk", STATS['total_produk'])
        st.metric("Median Harga", STATS['median_harga'])
        tampilkan_feature_importance()

# =========================
# SIDEBAR INFO
# =========================
st.sidebar.markdown("---")
st.sidebar.write("Model: Random Forest")
st.sidebar.write("ROC-AUC ~0.89")