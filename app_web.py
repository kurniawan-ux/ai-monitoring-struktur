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
    st.markdown(
    '<p style="color:#FFFFFF; font-size:17px; font-weight:700;">👨‍💻 Kurniawan Ilyas</p>',
    unsafe_allow_html=True
)

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
st.markdown('<p class="main-title">🏗️ Dasbor AI Monitoring Gaya Dalam & Keandalan Struktur</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Sistem Peringatan Dini Keandalan Jembatan Berbasis AI</p>', unsafe_allow_html=True)


# 4. ENGINE BACKEND: LOADING DATA & TRAINING AI

# A. IMPORT PUSTAKA BACKEND
from pathlib import Path
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score


# B. MEMUAT DATA EXCEL
BASE_DIR = Path(__file__).resolve().parent
FILE_EXCEL = BASE_DIR / "data_sensor_jembatan.xlsx"

if not FILE_EXCEL.exists():
    st.error(f"File Excel tidak ditemukan: {FILE_EXCEL}")
    st.stop()

df = pd.read_excel(FILE_EXCEL)

tab_dashboard, tab_edukasi, tab_data, tab_analisis = st.tabs(
    [
        "📊 Dasbor Monitoring",
        "📖 Dokumentasi Edukasi",
        "📂 Basis Data Excel",
        "🏗️ Analisis Gaya Dalam & Keandalan Struktur",
    ]
)

# ============================================================
# 6. KONTEN FRONT-END SETIAP TAB
# ============================================================

# TAB 1: DASBOR MONITORING
with tab_dashboard:
    st.subheader("Analisis Gaya Dalam dan Keandalan Struktur Jembatan")
    st.caption(
        "Dashboard ini menggunakan data dari data_sensor_jembatan.xlsx. "
        "Hasil evaluasi kapasitas memerlukan parameter desain yang terverifikasi."
    )

    st.metric("Jumlah baris data Excel", f"{len(df):,}")
    st.metric("Jumlah kolom data", f"{len(df.columns):,}")

    st.markdown("### Grafik respons data sensor")

    # Pilih kolom numerik yang tersedia pada Excel
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if numeric_cols:
        selected_cols = st.multiselect(
            "Pilih variabel untuk ditampilkan",
            options=numeric_cols,
            default=numeric_cols[:min(3, len(numeric_cols))]
        )

        if selected_cols:
            st.line_chart(df[selected_cols])
        else:
            st.info("Pilih minimal satu variabel untuk menampilkan grafik.")
    else:
        st.warning("Tidak ditemukan kolom numerik pada file Excel.")

    st.caption(
        "Sumbu horizontal menunjukkan urutan baris Excel. "
        "Sumbu tersebut bukan waktu aktual kecuali tersedia kolom waktu."
    )

    st.markdown("### Pemeriksaan awal data")
    st.write("Jumlah data kosong per kolom:")
    st.dataframe(
        df.isna().sum().rename("Jumlah data kosong").to_frame(),
        use_container_width=True
    )


# TAB 2: DOKUMENTASI EDUKASI
with tab_edukasi:
    st.subheader("Dokumentasi Edukasi Teknik Struktur")

    st.markdown("""
    **Materi pemeriksaan struktur jembatan**

    1. Statika dan mekanika rekayasa.
    2. Reaksi perletakan dan keseimbangan struktur.
    3. Diagram gaya geser dan momen lentur.
    4. Pemeriksaan kapasitas lentur dan geser.
    5. Interpretasi respons sensor dan keterbatasan data.
    6. Verifikasi terhadap gambar, spesifikasi, dan standar desain
       yang berlaku.
    """)


# TAB 3: BASIS DATA EXCEL
with tab_data:
    st.subheader("Basis Data Sensor Jembatan")

    st.write(f"Nama file: {FILE_EXCEL.name}")
    st.write(f"Jumlah baris: {len(df):,}")
    st.write(f"Jumlah kolom: {len(df.columns):,}")

    st.dataframe(df, use_container_width=True)

    st.download_button(
        label="Unduh data Excel",
        data=FILE_EXCEL.read_bytes(),
        file_name=FILE_EXCEL.name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    
# TAB 4: ANALISIS GAYA DALAM DAN KEANDALAN STRUKTUR
with tab_analisis:
    st.header("Analisis Gaya Dalam & Keandalan Struktur Jembatan")

    st.write("Sumber data:", FILE_EXCEL.name)
    st.write("Jumlah baris data sensor:", len(df))

    st.subheader("Data sensor jembatan")
    st.dataframe(df, use_container_width=True)

    st.info(
        "Data sensor ditampilkan untuk pemeriksaan awal. "
        "Penilaian keandalan struktur memerlukan verifikasi "
        "model, beban, kapasitas, dan validitas data sensor."
    )


    # TANYA JAWAB UMUM TEKNIK SIPIL
    st.divider()
    st.subheader("💬 Tanya Jawab Umum Teknik Sipil")
    st.caption(
        "Topik: struktur, beton, baja, geoteknik, jalan, "
        "jembatan, material, pengujian, dan manajemen konstruksi."
    )

    pertanyaan = st.text_area(
        "Tuliskan pertanyaan Teknik Sipil Anda:",
        placeholder=(
            "Contoh: Apa dasar pemeriksaan lentur balok beton "
            "bertulang berdasarkan SNI?"
        ),
        key="pertanyaan_teknik_sipil",
        height=100,
    )

    if st.button("Cari Jawaban", key="btn_tanya_teknik_sipil"):
        if not pertanyaan.strip():
            st.warning("Masukkan pertanyaan terlebih dahulu.")
        else:
            p = pertanyaan.lower()

            referensi = [
                {
                    "kata": ["beton bertulang", "balok beton", "kolom beton"],
                    "topik": "Struktur beton bertulang",
                    "jawaban": (
                        "Pemeriksaan umumnya mencakup kekuatan lentur, "
                        "geser, aksial, penyaluran tulangan, dan "
                        "kemampuan layan. Parameter desain dan "
                        "ketentuan detail harus diperiksa terhadap "
                        "standar yang berlaku serta kondisi proyek."
                    ),
                    "standar": "SNI 2847:2019; ACI 318 sebagai referensi terkait."
                },
                {
                    "kata": ["beban", "beban hidup", "beban mati", "angin"],
                    "topik": "Beban dan kombinasi pembebanan",
                    "jawaban": (
                        "Identifikasi beban mati, beban hidup, angin, "
                        "gempa, dan beban lain yang relevan. Tentukan "
                        "kombinasi beban sesuai fungsi bangunan dan "
                        "ketentuan desain yang berlaku."
                    ),
                    "standar": "SNI 1727:2020; SNI 1726:2019 untuk aspek gempa."
                },
                {
                    "kata": ["baja", "profil wf", "struktur baja"],
                    "topik": "Struktur baja",
                    "jawaban": (
                        "Pemeriksaan dapat meliputi tarik, tekan, "
                        "lentur, geser, tekuk lokal, tekuk lateral, "
                        "sambungan, dan kondisi layan. Verifikasi "
                        "mutu material, dimensi, tumpuan, serta "
                        "kombinasi pembebanan."
                    ),
                    "standar": "SNI 1729:2020; AISC 360 sebagai referensi terkait."
                },
                {
                    "kata": ["tanah", "pondasi", "fondasi", "sondir", "cpt"],
                    "topik": "Geoteknik dan fondasi",
                    "jawaban": (
                        "Analisis bergantung pada stratigrafi tanah, "
                        "hasil investigasi, daya dukung, penurunan, "
                        "air tanah, dan kondisi konstruksi. Data "
                        "sondir saja tidak selalu cukup untuk seluruh "
                        "parameter desain."
                    ),
                    "standar": "SNI 8460:2017; standar pengujian tanah ASTM yang relevan."
                },
                {
                    "kata": ["agregat", "semen", "kuat tekan beton", "silinder"],
                    "topik": "Material dan pengujian beton",
                    "jawaban": (
                        "Pastikan prosedur pengambilan sampel, "
                        "pembuatan benda uji, perawatan, umur uji, "
                        "dan pengujian mengikuti metode yang sesuai. "
                        "Hasil harus dibandingkan dengan kriteria "
                        "penerimaan yang berlaku."
                    ),
                    "standar": "SNI terkait beton dan pengujian; ASTM C39/C39M untuk kuat tekan silinder."
                },
            ]

            hasil = [
                r for r in referensi
                if any(k in p for k in r["kata"])
            ]

            if hasil:
                for r in hasil:
                    st.markdown(f"### {r['topik']}")
                    st.write(r["jawaban"])
                    st.caption(
                        "Referensi awal: " + r["standar"]
                    )
            else:
                st.info(
                    "Belum ditemukan jawaban pada basis pengetahuan "
                    "awal. Pertanyaan perlu ditelaah menggunakan "
                    "standar resmi yang relevan."
                )

            st.warning(
                "Verifikasi nomor, edisi, status berlaku, amendemen, "
                "dan pasal standar pada sumber resmi sebelum "
                "digunakan untuk keputusan desain atau pemeriksaan."
            )
