import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. Konfigurasi Awal Halaman Web Dasbor
st.set_page_config(page_title="Dasbor Monitoring AI Sipil", layout="wide")
st.title("🏗️ Dasbor AI Monitoring Gaya Dalam & Kerusakan Struktur")
st.markdown("### 🧑‍💻 Karya Kurniawan Ilyas - Analis AI Engineer Structure")
st.image("foto_kurniawan.jpg.jpg", width=150, caption="Dr.Kurniawan Ilyas, S.T.")
st.write("Aplikasi visualisasi monitoring nilai Momen Aktual berdasarkan input Sensor Regangan.")

# 2. Pemuatan Berkas Basis Data Excel Lapangan
@st.cache_data
def load_data():
    return pd.read_excel("data_sensor_jembatan.xlsx")

df = load_data()

# 3. Proses Latih Algoritma Jaringan Saraf Tiruan AI
X = df[['Sensor_Tumpuan_Kiri', 'Sensor_Tengah_Bentang', 'Sensor_Tumpuan_Kanan']]
y = df['Momen_Aktif_kNm']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ai_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
ai_model.fit(X_train_scaled, y_train)

# Prediksi Data Uji
predictions = ai_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

st.markdown("---")

# 4. TAMPILAN PANEL PANEL UTAMA DI WEB LOCALHOST
col1, col2 = st.columns(2)
with col1:
    st.metric(label="📊 Akurasi Regresi Model (R2 Score)", value=f"{r2*100:.2f} %")
with col2:
    st.metric(label="📉 Rata-rata Nilai Eror (MSE)", value=f"{mse:.4f}")

st.markdown("---")

# 5. PANEL INTERAKTIF INPUT SENSOR LAPANGAN RIIL
st.subheader("🔮 Uji Coba Input Nilai Sensor Lapangan Baru")
input_kiri = st.number_input("Sensor Tumpuan Kiri (microstrain)", min_value=0.0, max_value=1000.0, value=250.0)
input_tengah = st.number_input("Sensor Tengah Bentang (microstrain)", min_value=0.0, max_value=1000.0, value=400.0)
input_kanan = st.number_input("Sensor Tumpuan Kanan (microstrain)", min_value=0.0, max_value=1000.0, value=150.0)

# Konversi Prediksi Nilai Baru dari Input Pengguna
data_sensor_baru = np.array([[input_kiri, input_tengah, input_kanan]])
data_sensor_scaled = scaler.transform(data_sensor_baru)
prediksi_momen = ai_model.predict(data_sensor_scaled)

# Tampilkan Hasil Estimasi Momen dalam Bentuk Banner Sukses Hijau
st.success(f"### 📈 Hasil Estimasi Gaya Momen Dalam Struktur: **{prediksi_momen[0]:.2f} kNm**")

st.markdown("---")

# 6. MENCETAK GRAFIK VISUALISASI SCATTER PLOT DI WEB
st.subheader("📉 Grafik Print Out Analisis Akurasi AI")
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.scatter(y_test, predictions, color='blue', alpha=0.5, label='Prediksi AI')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, label='Garis Ideal')
ax.set_xlabel('Momen Aktual dari Excel (kNm)')
ax.set_ylabel('Momen Hasil Prediksi AI (kNm)')
ax.legend()
ax.grid(True)

# Masukkan objek gambar grafik ke dalam halaman web streamlit
st.pyplot(fig)
