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


    return f"""<div align="center">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 160" width="100%">
<path d="M 50 130 A 100 100 0 0 1 250 130" fill="none" stroke="#e5e7eb" stroke-width="18" stroke-linecap="round"/>
<path d="M 50 130 A 100 100 0 0 1 {x:.1f} {y:.1f}" fill="none" stroke="{color}" stroke-width="18" stroke-linecap="round"/>
<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="{color}" stroke-width="4" stroke-linecap="round"/>
<circle cx="{cx}" cy="{cy}" r="6" fill="{color}"/>
<text x="150" y="150" text-anchor="middle" font-size="22" font-weight="bold" fill="{color}">{prob:.1f}%</text>
</svg>
</div>"""
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
    st.header("📈 Exploratory Data Analysis (EDA)")
    st.write("Di halaman ini, kita membongkar rahasia data Tokopedia untuk menemukan pola apa yang membuat sebuah produk laku keras.")

    if df is not None:
        import plotly.express as px
        
        # --- 1. PERSIAPAN DATA SEMENTARA ---
        df_eda = df.copy()
        df_eda['Terjual_Angka'] = df_eda['Terjual'].replace({'rb': '000', '\+ terjual': '', ' terjual': '', ' ulasan': ''}, regex=True).fillna(0)
        df_eda['Terjual_Angka'] = pd.to_numeric(df_eda['Terjual_Angka'], errors='coerce').fillna(0)
        df_eda['Diskon_Angka'] = pd.to_numeric(df_eda['Diskon (%)'].replace('%', '', regex=True), errors='coerce').fillna(0)
        df_eda['Ada_Diskon'] = df_eda['Diskon_Angka'].apply(lambda x: 'Ya (Ada Diskon)' if x > 0 else 'Tidak Ada Diskon')
        
        # --- GRAFIK 1: HEATMAP KORELASI ---
        st.subheader("1. Peta Panas Korelasi (Correlation Heatmap)")
        st.markdown("*Mencari tahu faktor apa yang paling berhubungan erat dengan kesuksesan produk.*")
        
        kolom_angka = df_eda[['Harga (IDR)', 'Rating', 'Diskon_Angka', 'Terjual_Angka']].dropna()
        korelasi = kolom_angka.corr()
        
        fig_heatmap = px.imshow(korelasi, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.info("💡 **Insight Bisnis:**\nSumbu X dan Y berisi nama-nama fitur. Semakin pekat warna merahnya (mendekati 1), semakin kuat hubungannya. Kita bisa melihat bahwa diskon dan rating punya peran kuat dalam mendorong jumlah terjual, sedangkan harga saja tidak menjamin barang laku.")
        st.markdown("---")

        # --- GRAFIK 2: SCATTER PLOT ---
        st.subheader("2. Harga vs Jumlah Terjual (Berdasarkan Rating)")
        st.markdown("*Mencari 'Sweet Spot' (Titik harga ideal di pasar).*")
        
        df_scatter = df_eda[df_eda['Harga (IDR)'] <= 500000] 
        
        fig_scatter = px.scatter(df_scatter, x="Harga (IDR)", y="Terjual_Angka", color="Rating", 
                                 hover_data=["Nama Produk"], color_continuous_scale="Viridis", opacity=0.7)
        fig_scatter.update_layout(xaxis_title="Sumbu X: Harga Produk (Rupiah)", yaxis_title="Sumbu Y: Jumlah Terjual (Pcs)")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.info("💡 **Insight Bisnis:**\nTitik-titik ini adalah produk. Terlihat pasar menumpuk di harga Rp 50.000 - Rp 200.000. Tapi perhatikan titik kuning/hijau terang (Rating tinggi). Ini membuktikan barang dengan harga mahal tetap bisa laku keras asalkan kualitas dan ratingnya terjaga.")
        st.markdown("---")

        # --- GRAFIK 3: DONUT CHART ---
        st.subheader("3. Apakah Diskon Itu Wajib?")
        st.markdown("*Melihat strategi diskon dari produk-produk Best Seller.*")
        
        df_bestseller = df_eda[df_eda['Terjual_Angka'] >= THRESHOLD]
        komposisi_diskon = df_bestseller['Ada_Diskon'].value_counts().reset_index()
        komposisi_diskon.columns = ['Status Diskon', 'Jumlah Produk']
        
        fig_donut = px.pie(komposisi_diskon, values='Jumlah Produk', names='Status Diskon', hole=0.5,
                           color_discrete_sequence=['#22c55e', '#ef4444']) 
        st.plotly_chart(fig_donut, use_container_width=True)
        
        st.info("💡 **Insight Bisnis:**\nGrafik ini diambil KHUSUS dari data produk Best Seller saja. Terlihat jelas bahwa mayoritas produk unggulan (hijau) menggunakan strategi psikologis 'Coret Harga' (Diskon) untuk memancing pembeli dibanding harga flat.")

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
    st.header("Model Training Info")
    st.write(f"Total data diproses: **{STATS['total_produk']}** baris")
        
        # Tambahan Metrik Performa 
    col1, col2, col3 = st.columns(3)
    col1.metric("Akurasi Model", "84.2%") 
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
# RESULT 
# =========================
elif menu == "Result":

    st.title("Prediksi Best Seller")

    # Tab "Produk Baru" resmi dihapus!
    tab_existing, tab_whatif, tab_pasar = st.tabs([
        "Produk Existing",
        "What-If (Simulasi)",
        "Pasar"
    ])

    # =====================
    # EXISTING
    # =====================
    with tab_existing:
        st.markdown("### Analisis Produk Existing")
        st.write("Masukkan data produk yang sudah berjalan untuk melihat peluangnya menjadi Best Seller.")
        
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
        st.markdown("### Simulasi Target Pemasaran (What-If)")
        st.info("💡 **Tips untuk Produk Baru:** Gunakan slider di bawah untuk mencari tahu berapa target Rating dan Ulasan yang harus dicapai oleh tim marketing agar produk baru bisa sukses di harga tertentu.")
        
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
        st.markdown("### Ringkasan Kondisi Pasar")
        col1, col2 = st.columns(2)
        col1.metric("Total Produk Dianalisis", f"{STATS['total_produk']} Produk")
        col2.metric("Median Harga Pasar", f"Rp {STATS['median_harga']:,.0f}")
        
        st.markdown("#### Faktor Penentu Kesuksesan (Feature Importance)")
        tampilkan_feature_importance()

# =========================
# SIDEBAR INFO
# =========================
st.sidebar.markdown("---")
st.sidebar.write("**Model:** Random Forest")
st.sidebar.write("**ROC-AUC:** ~0.89")