import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. KONFIGURASI FRONT-END UTAMA (TEMA PREMIUM & ELEGAN)
st.set_page_config(page_title="AI Structural Monitoring Dashboard", layout="wide", initial_sidebar_state="expanded")

# DESAIN CSS ADVANCED: Latar Belakang Gradasi Teknik & Kartu Metrik Kelas Industri
st.markdown("""
    <style>
    /* Mengubah Latar Belakang Utama Aplikasi Web */
    .stApp {
        background: linear-gradient(135deg, #F0F4F8 0%, #E2E8F0 100%);
    }
    
    /* Pengaturan Tipografi Judul Utama */
    .main-title { 
        font-size: 38px !important; 
        font-weight: 800; 
        color: #1E3A8A; 
        text-align: left; 
        margin-bottom: 5px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .sub-title { 
        font-size: 18px !important; 
        color: #475569; 
        font-weight: 500; 
        margin-bottom: 25px; 
    }
    
    /* Desain Kartu Indikator Metrik Keren Bersayap & Bayangan Halus */
    .card-metric { 
        background: white; 
        padding: 22px; 
        border-radius: 12px; 
        border-top: 4px solid #3B82F6; 
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
        transition: transform 0.2s;
    }
    .card-metric:hover {
        transform: translateY(-3px);
    }
    
    /* Styling Sidebar */
    .css-12rw0g6, .stSidebar {
        background-color: #0F172A !important;
    }
    </style>
""", unsafe_allow_html=True)

# 2. SIDEBAR KIRI: PROFIL PENGEMBANG & INFORMASI TEKNIS
with st.sidebar:
    st.image("foto_kurniawan.jpg.jpg", width=180)
    st.markdown("### 🧑‍💻 **Kurniawan Ilyas**")
    st.markdown("<p style='color:#94A3B8; font-size:14px; margin-top:-10px;'>Analis AI Engineer Structure<br>Magister Teknik Sipil</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📚 **Informasi Edukasi**")
    st.info("""
    **Metode AI:** *Multi-Layer Perceptron (MLP)* Jaringan Saraf Tiruan.
    
    **Konsep Mekanika:** AI bertindak sebagai *Virtual Sensor* untuk memetakan hubungan non-linear antara regangan material (*microstrain*) akibat beban lalu lintas dengan respons gaya dalam momen lentur (**kNm**).
    """)
    st.markdown("---")
    st.markdown("<p style='text-align:center; color:#64748B; font-size:12px;'>SHM AI System v2.0</p>", unsafe_allow_html=True)

# 3. KONTEN UTAMA: HEADER
st.markdown('<p class="main-title">🏗️ Dasbor AI Monitoring Gaya Dalam & Kesehatan Struktur</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Sistem Peringatan Dini Kesehatan Jembatan Berbasis Sensor IoT & Jaringan Saraf Tiruan</p>', unsafe_allow_html=True)

# 4. ENGINE BACKEND: LOADING DATA & TRAINING AI
      # Penentuan Indikator Status Batas Keamanan Struktur (Safety Threshold)
        if prediksi_momen > 500:
            st.error(f"🚨 STATUS: BAHAYA (CRITICAL RISK)")
            st.info(f"Estimasi Momen Tengah: {prediksi_momen:.2f} kNm. Struktur jembatan telah melewati Batas Ultra (Ultimate Limit State)!")
        elif prediksi_momen > 300:
            st.warning(f"⚠️ STATUS: WASPADA (WARNING)")
            st.info(f"Estimasi Momen Tengah: {prediksi_momen:.2f} kNm. Tegangan struktur mendekati Batas Layan (Serviceability Limit State).")
        else:
            st.success(f"✅ STATUS: AMAN (NORMAL)")
            st.info(f"Estimasi Momen Tengah: {prediksi_momen:.2f} kNm. Struktur jembatan beroperasi penuh dalam kondisi elastis aman.")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ai_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
ai_model.fit(X_train_scaled, y_train)

predictions = ai_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# 5. STRUKTUR FRONT-END UTAMA (TAB LAYOUT INTERAKTIF)
tab_dashboard, tab_edukasi, tab_data = st.tabs(["📊 Dasbor Monitoring", "📖 Dokumentasi Edukasi", "💾 Basis Data Excel"])

with tab_dashboard:
    # Baris 1: Panel Indikator Utama (Metrik Kinerja AI)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"<div class='card-metric'><b>📊 Akurasi Jaringan AI (R2 Score)</b><br><span style='font-size:28px; font-weight:bold; color:#10B981;'>{r2*100:.2f} %</span></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='card-metric'><b>📉 Rata-rata Nilai Eror (MSE)</b><br><span style='font-size:28px; font-weight:bold; color:#EF4444;'>{mse:.4f}</span></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='card-metric'><b>🗃️ Kapasitas Data Latih</b><br><span style='font-size:28px; font-weight:bold; color:#1E3A8A;'>{len(df)} Sampel</span></div>", unsafe_allow_html=True)

    st.markdown("###")

    # Baris 2: Pemisahan 2 Kolom Seimbang (Kiri: Kendali Slider, Kanan: Grafik Proporsional)
    left_col, right_col = st.columns([1.1, 0.9])

    with left_col:
        st.markdown("#### 🔮 **Panel Kendali Nilai Sensor Real-Time**")
        st.caption("Geser slider di bawah untuk mensimulasikan beban dinamis kendaraan pada jembatan:")
        
        input_kiri = st.slider("Sensor Tumpuan Kiri (microstrain)", 0.0, 1000.0, 250.0)
        input_tengah = st.slider("Sensor Tengah Bentang (microstrain)", 0.0, 1000.0, 400.0)
        input_kanan = st.slider("Sensor Tumpuan Kanan (microstrain)", 0.0, 1000.0, 150.0)
        
        # Proses Prediksi Nilai Baru (Menggunakan Perbaikan Indeks Tunggal)
        data_sensor_baru = np.array([[input_kiri, input_tengah, input_kanan]])
        data_sensor_scaled = scaler.transform(data_sensor_baru)
        prediksi_momen = ai_model.predict(data_sensor_scaled)[0]
        
        # Penentuan Indikator Status Batas Keamanan Struktur (Safety Threshold)
        if prediksi_momen > 500:
            st.error(f"### 🚨 STATUS: BAHAYA (CRITICAL RISK)<br>Estimasi Momen Tengah: **{prediksi_momen:.2f} kNm**<br>Struktur jembatan mengalami deformasi kritis!")
        elif prediksi_momen > 300:
            st.warning(f"### ⚠️ STATUS: WASPADA (WARNING)<br>Estimasi Momen Tengah: **{prediksi_momen:.2f} kNm**<br>Beban struktur mendekati batas layan izin.")
        else:
            st.success(f"### ✅ STATUS: AMAN (NORMAL)<br>Estimasi Momen Tengah: **{prediksi_momen:.2f} kNm**<br>Struktur jembatan beroperasi dalam batas aman.")

        # Tombol Unduh Laporan (.txt)
        st.markdown("###")
        st.download_button(
            label="📥 Unduh Laporan Teknis Hasil Prediksi (.txt)",
            data=f"LAPORAN TEKNIS EXECUTIVE SUMMARY MONITORING STRUKTUR\nPengembang: Kurniawan Ilyas, S.T.\n\nHasil Pembacaan:\n- Sensor Kiri: {input_kiri} microstrain\n- Sensor Tengah: {input_tengah} microstrain\n- Sensor Kanan: {input_kanan} microstrain\n\nESTIMASI GAYA MOMEN DALAM: {prediksi_momen:.2f} kNm\nAkurasi Pemetaan AI: {r2*100:.2f}%",
            file_name="Laporan_Monitoring_AI.txt",
            mime="text/plain"
        )

    with right_col:
        st.markdown("#### 📉 **Grafik Validasi Akurasi Pemetaan AI**")
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.scatter(y_test, predictions, color='#2563EB', alpha=0.35, label='Prediksi Model MLP')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='#EF4444', lw=2, linestyle='--', label='Garis Ideal Lapangan')
        ax.set_xlabel('Momen Aktual Dokumen Excel (kNm)', fontsize=9)
        ax.set_ylabel('Momen Hasil Prediksi AI (kNm)', fontsize=9)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, linestyle=':', alpha=0.5)
        
        # PENGUNCI UTAMA: Mengunci grafik agar kompak, proporsional, dan pas di layar
        st.pyplot(fig, use_container_width=False)

with tab_edukasi:
    st.markdown("### 📘 **Dasar Teori Structural Health Monitoring (SHM)**")
    st.write("""
    Dalam rekayasa struktur jembatan, penentuan **gaya dalam (momen/geser)** secara konvensional memerlukan perhitungan matriks kekakuan elemen yang rumit. 
    Dengan pendekatan **Artificial Intelligence (AI)**, pola deformasi material yang ditangkap oleh sensor *strain gauge* dapat langsung dikonversi menjadi parameter momen lentur secara instan.
    
    Sistem monitoring ini dibagi menjadi dua bagian esensial:
    1. **Fase Deep Learning Pattern:** Jaringan Saraf Tiruan mempelajari korelasi regangan serat atas/bawah baja pada balok girder.
    2. **Fase Inferensi Real-Time:** Memberikan hasil tebakan keselamatan dalam hitungan milidetik saat kendaraan melintas di atas jembatan.
    """)
    st.image("https://unsplash.com", caption="Ilustrasi Implementasi Sensor Pintar pada Infrastruktur Sipil Modern")

with tab_data:
    st.markdown("#### 💾 **Dataset Mentah Pembacaan Sensor Jembatan**")
    st.caption("Menampilkan 100 baris data pertama dari file Excel 'data_sensor_jembatan.xlsx' untuk audit sistem:")
    st.dataframe(df.head(100), use_container_width=True)
