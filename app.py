import streamlit as st
import pandas as pd
import pickle

# Konfigurasi Halaman

st.set_page_config(
    page_title="Prediksi Best Seller",
    page_icon="🛍️",
    layout="wide"  
)

# Load model + Error handling
@st.cache_resource
def load_model():
    try:
        with open("model_bestseller.pickle", "rb") as file:
            package = pickle.load(file)
        return package
    except FileNotFoundError:
        return None
    
model_package = load_model()

# Validasi model berhasil di load
if model_package is None:
    st.error("WWWOOOPP... File 'model_bestseller.pickle' tidak ditemukan! Pastikan kamu sudah jalankan main_refactored.py terlebih dahulu. ")
    st.stop()


model = model_package['model']
FEATURES = model_package['features']
THRESHOLD = model_package['threshold']


def tampilkan_hasil(prediksi, probabilitas, harga, diskon, rating, ulasan):
    st.markdown("---")
    st.subheader("Hasil Prediksi")

    col_hasil, col_prob = st.columns([2, 1])

    with col_hasil:
        if prediksi == 1:
            st.success(f"**WEDEHH, BERPOTENSI BEST SELLER!**")
            st.balloons()
        else:
            st.warning(f"**KURANG BERPOTENSI BEST SELLER EUY**")
    with col_prob:
        st.metric("Pelung Best Seller:", f"{probabilitas:.1f}")

    st.progress(int(probabilitas))

    # Pemberian saran
    st.markdown('#### Saran Sesifik')
    saran = []

    if diskon == 0:
        saran.append("- **Coba tambahkan diskon** (minimal 10-20%) untuk menarik perhatian pembeli.")
    if harga > 500_000 and diskon < 20:
        saran.append("- Harga di atas Rp 500.000 butuh **diskon lebih besar** agar kompetitif.")
    if rating < 4.0 and rating > 0:
        saran.append("- **Tingkatkan kualitas produk & layanan** untuk mendongkrak rating di atas 4.0.")
    if ulasan < 10 and ulasan > 0:
        saran.append("- **Dorong pembeli untuk meninggalkan ulasan** (mis. dengan bonus kecil atau pesan follow-up).")
    if probabilitas >= 70:
        saran.append("- Strategi produkmu sudah bagus! Fokus pada **konsistensi stok dan kecepatan pengiriman**.")

    if saran:
        for s in saran:
            st.markdown(s)
    else:
        st.markdown("- Coba optimalkan kombinasi harga dan diskon untuk meningkatkan daya saing.")

    # Tabel Ringkasan
    with st.expander("🔍 Detail Input yang Dianalisis"):
        ringkasan = {
            "Parameter": ["Harga", "Diskon", "Harga Efektif", "Rating", "Jumlah Ulasan"],
            "Nilai": [
                f"Rp {harga:,.0f}",
                f"{diskon}%",
                f"Rp {harga * (1 - diskon/100):,.0f}",
                f"{rating}",
                f"{ulasan} ulasan"
            ]
        }
        st.table(pd.DataFrame(ringkasan))


# HEADER
st.title("🛍️ Prediksi Potensi Best Seller Produk")
st.write("Masukkan detail rencana produkmu di bawah ini, dan AI akan memprediksi seberapa besar peluang produk tersebut laku keras di pasaran.")

st.markdown("---")

tab1, tab2 = st.tabs(["Produk Baru (Belum Launch)", "Produk Existing (Sudah Punya Data)"])

with tab1:
    st.subheader("Simulasi Produk Baru")
    st.caption("Untuk produk yang belum punya rating & ulasan. Rating & ulasan akan diasumsikan 0.")

    col1, col2 = st.columns(2)

    with col1:
        harga_baru = st.number_input("Harga Produk (Rp)", min_value=0, value=99000, step=5000, key="harga_baru")
        diskon_baru = st.slider("Rencana diskon (%)", min_value=0, max_value=90, step=5, key="diskon_baru")

    with col2:
        # Tampilkan harga setelah diskon secara real time
        harga_efektif_baru = harga_baru * (1 - diskon_baru / 100)
        st.metric("Harga Setelah Diskon", f"Rp {harga_efektif_baru:,.0f}")
        st.caption("Rating & ulasan diasumsikan 0 karena produk baru")

    # Validasi input harga
    if harga_baru == 0:
        st.warning("HARGA 0 TIDAK MASUK AKAL!! Masukkan harga produk yang valid")

# Tombol
    if st.button("Prediksi Potensi Best Sellers!", use_container_width=True, key="btn_baru"):
        # Bungkus inputan user ke bentuk tabel
        input_dict = {
            'Harga (IDR)': harga_baru,
            'Diskon (%)': diskon_baru,
            'Rating': 0.0,
            'Ulasan_bersih': 0,
            'Harga_setelah_diskon': harga_efektif_baru,
            'Ada_diskon': 1 if diskon_baru > 0 else 0,
            'Skor_kepercayaan': 0.0,
        }
        data_input = pd.DataFrame([input_dict])[FEATURES]

        # Mesin melakukan tebakan
        prediksi =  model.predict(data_input)[0]
        probabilitas = model.predict_proba(data_input)[0][1] * 100

        # Menampilkan hasil
        tampilkan_hasil(prediksi, probabilitas, harga_baru, diskon_baru, 0.0, 0)

with tab2:
    st.subheader("Analisis Produk")
    st.caption("Untuk produk yang sudah berjalan dan punya data penjualan nyata.")
    col_ex1, col_ex2 = st.columns(2)

    with col_ex1:
        harga_ex = st.number_input("Harga Produk (Rp)", min_value=0, value=99000, step=5000, key="harga_ex")
        diskon_ex = st.slider("Diskon (%)", min_value=0, max_value=90, step=5, key="diskon_ex")
    with col_ex2:
        rating_ex = st.slider("Rating Aktual (1.0 - 5.0)", min_value=0.0, max_value=5.0, value=4.5, step=0.1, key="rating_ex")
        ulasan_ex = st.number_input("Jumlah Ulasan Aktual", min_value=0, value=50, step=10, key="ulasan_ex")
 
    harga_efektif_ex = harga_ex * (1 - diskon_ex / 100)
 
    if st.button("Prediksi Produk Existing", use_container_width=True, key="btn_ex"):
        input_dict = {
            'Harga (IDR)': harga_ex,
            'Diskon (%)': diskon_ex,
            'Rating': rating_ex,
            'Ulasan_bersih': ulasan_ex,
            'Harga_setelah_diskon': harga_efektif_ex,
            'Ada_diskon': 1 if diskon_ex > 0 else 0,
            'Skor_kepercayaan': rating_ex * ulasan_ex,  
        }
        data_input = pd.DataFrame([input_dict])[FEATURES]
 
        prediksi = model.predict(data_input)[0]
        probabilitas = model.predict_proba(data_input)[0][1] * 100
 
        tampilkan_hasil(prediksi, probabilitas, harga_ex, diskon_ex, rating_ex, ulasan_ex)

with st.sidebar:
    st.header("ℹ️ Tentang Model")
    st.write(f"""
    **Algoritma**: Random Forest Classifier
    
    **Fitur yang digunakan**:
    - Harga produk
    - Diskon
    - Rating
    - Jumlah ulasan
    - Harga setelah diskon 
    - Ada/tidak diskon 
    - Skor kepercayaan 
    
    **Definisi Best Seller**: Produk dengan penjualan > {THRESHOLD:,.0f} pcs (persentil 75 data)
    
    **Catatan**: Model ini dilatih dari data produk Tokopedia. Prediksi bersifat estimasi dan tidak menjamin hasil aktual.
    """)

 
    st.markdown("---")
    st.caption("Dibuat dengan ❤️ azek")

