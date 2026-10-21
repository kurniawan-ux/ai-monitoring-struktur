import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os
from openai import OpenAI

# 1. KONFIGURASI FRONT-END UTAMA (TEMA PREMIUM & ELEGAN)
st.set_page_config(page_title="AI Structural Monitoring Dashboard", layout="wide", initial_sidebar_state="expanded")

def jawab_lokal(pertanyaan):
    """Jawaban dasar Teknik Sipil tanpa OpenAI API."""
    q = pertanyaan.lower()

    if "apa itu beton" in q or "pengertian beton" in q:
        return (
            "Beton adalah material komposit yang umumnya terdiri atas "
            "semen hidraulis, air, agregat halus, dan agregat kasar. "
            "Kinerja beton dipengaruhi oleh proporsi campuran, kualitas "
            "bahan, pemadatan, dan perawatan (curing). Gunakan standar "
            "yang relevan dan data hasil pengujian material."
        )

    if "reaksi tumpuan" in q:
        return (
            "Reaksi tumpuan dihitung menggunakan keseimbangan statika: "
            "ΣFx = 0, ΣFy = 0, dan ΣM = 0. Tumpuan sendi menahan gaya "
            "horizontal dan vertikal; rol menahan satu arah gaya; "
            "jepit menahan gaya horizontal, vertikal, dan momen. "
            "Perhitungan numerik memerlukan skema struktur dan beban."
        )

    if "gaya geser" in q or "momen lentur" in q:
        return (
            "Gaya geser dan momen lentur ditentukan dari keseimbangan "
            "struktur, beban, panjang bentang, dan kondisi tumpuan. "
            "Masukkan data struktur untuk memperoleh perhitungan numerik."
        )

    if "geoteknik" in q or "daya dukung tanah" in q:
        return (
            "Analisis geoteknik memerlukan data tanah, stratigrafi, "
            "muka air tanah, beban fondasi, dimensi fondasi, dan "
            "parameter tanah. Daya dukung serta penurunan harus "
            "diperiksa sesuai kondisi proyek dan standar yang berlaku."
        )

    return (
        "Pertanyaan belum tersedia dalam materi jawaban lokal. "
        "Coba kata kunci: apa itu beton, reaksi tumpuan, gaya geser, "
        "momen lentur, atau daya dukung tanah. Jawaban ini merupakan "
        "penjelasan awal, bukan verifikasi desain struktur."
    )

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


# FORM INPUT DATA SENSOR
st.markdown("### Input Data Sensor")

with st.form("form_input_sensor"):
    col1, col2 = st.columns(2)

    with col1:
        sensor_kiri = st.number_input(
            "Sensor Tumpuan Kiri (microstrain)",
            value=0.0,
            format="%.3f"
        )
        sensor_tengah = st.number_input(
            "Sensor Tengah Bentang (microstrain)",
            value=0.0,
            format="%.3f"
        )

    with col2:
        sensor_kanan = st.number_input(
            "Sensor Tumpuan Kanan (microstrain)",
            value=0.0,
            format="%.3f"
        )
        momen_aktif = st.number_input(
            "Momen Aktif (kN·m)",
            value=0.0,
            format="%.3f"
        )

    waktu_data = st.text_input(
        "Waktu pengukuran",
        placeholder="Contoh: 2026-10-21 14:30"
    )

    simpan = st.form_submit_button("Simpan Data Sensor")

if simpan:
    data_baru = {
        "Waktu Pengukuran": waktu_data,
        "Sensor_Tumpuan_Kiri": sensor_kiri,
        "Sensor_Tengah_Bentang": sensor_tengah,
        "Sensor_Tumpuan_Kanan": sensor_kanan,
        "Momen_Aktif_kNm": momen_aktif
    }

    st.session_state.setdefault("data_input_sensor", [])
    st.session_state["data_input_sensor"].append(data_baru)

    st.success("Data sensor ditambahkan ke sesi aplikasi.")

if st.session_state.get("data_input_sensor"):
    st.markdown("#### Data Sensor yang Diinput")
    st.dataframe(
        pd.DataFrame(st.session_state["data_input_sensor"]),
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

    
    # =========================================================
    # TANYA JAWAB UMUM TEKNIK SIPIL BERBASIS AI
    # =========================================================

    st.divider()
    st.subheader("💬 Asisten AI Teknik Sipil")
    st.caption(
        "Tanyakan konsep, metode, analisis, perhitungan, "
        "material, geoteknik, struktur, konstruksi, dan standar."
    )

    # Ambil API key dari environment atau Streamlit Secrets
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            api_key = ""

    # Riwayat percakapan disimpan selama sesi berlangsung
    if "riwayat_tanya_teknik_sipil" not in st.session_state:
        st.session_state.riwayat_tanya_teknik_sipil = []

    # Tampilkan percakapan sebelumnya
    for pesan in st.session_state.riwayat_tanya_teknik_sipil:
        with st.chat_message(pesan["role"]):
            st.markdown(pesan["content"])

    # Form pertanyaan
    with st.form(
        "form_asisten_teknik_sipil",
        clear_on_submit=True
    ):
        pertanyaan_ai = st.text_area(
            "Tuliskan pertanyaan Teknik Sipil:",
            placeholder=(
                "Contoh:\n"
                "- Apa itu beton?\n"
                "- Hitung reaksi tumpuan balok sederhana.\n"
                "- Bagaimana pemeriksaan lentur balok beton?\n"
                "- Bagaimana analisis daya dukung fondasi?\n"
                "- Apa perbedaan metode LRFD dan ASD?"
            ),
            height=130,
            key="input_pertanyaan_ai",
        )

        kirim_pertanyaan = st.form_submit_button(
            "🚀 Tanya AI Teknik Sipil",
            use_container_width=True,
        )

    if kirim_pertanyaan:
        if not pertanyaan_ai.strip():
            st.warning("Masukkan pertanyaan terlebih dahulu.")

        elif not api_key:
            st.error(
                "API key belum dikonfigurasi. "
                "Ikuti petunjuk konfigurasi API key di bawah kode."
            )

        else:
            try:
                with st.spinner(
                    "AI sedang menganalisis pertanyaan Anda..."
                ):
                    client = OpenAI(api_key=api_key)

                    # Kirim riwayat terbatas agar percakapan
                    # sebelumnya tetap menjadi konteks.
                    riwayat = (
                        st.session_state.riwayat_tanya_teknik_sipil
                    )

                    input_ai = riwayat[-10:] + [
                        {
                            "role": "user",
                            "content": pertanyaan_ai.strip(),
                        }
                    ]

                    instruksi_ai = """
Anda adalah Asisten AI Teknik Sipil yang membantu
mahasiswa, insinyur, konsultan, pelaksana konstruksi,
pengawas, dan pemeriksa pekerjaan.

CAKUPAN:
1. Mekanika teknik dan analisis struktur.
2. Beton bertulang, beton prategang, dan struktur baja.
3. Geoteknik, penyelidikan tanah, dan fondasi.
4. Jalan, jembatan, drainase, hidrologi, dan sumber daya air.
5. Material, pengujian, kendali mutu, dan metode konstruksi.
6. RAB, AHSP, estimasi biaya, manajemen proyek, dan BIM.
7. Perhitungan teknik, rumus, asumsi, satuan, dan interpretasi hasil.
8. SNI, ACI, ASTM, AISC, serta ketentuan teknis terkait.

ATURAN JAWABAN:
- Jawab dalam Bahasa Indonesia yang jelas, sistematis,
  profesional, dan sesuai konteks pertanyaan.
- Jawab langsung pertanyaan pengguna, termasuk pertanyaan
  dasar seperti "Apa itu beton?".
- Untuk perhitungan, jelaskan data diketahui, asumsi,
  rumus, substitusi angka, satuan, hasil, dan pemeriksaan
  kewajaran hasil bila relevan.
- Jika data tidak cukup, nyatakan data yang kurang.
  Jangan mengarang dimensi, beban, mutu material, atau hasil uji.
- Bedakan fakta, asumsi, pendekatan awal, dan kesimpulan.
- Jangan mengarang nomor pasal, judul standar, kutipan,
  status berlaku, edisi, atau hasil verifikasi.
- Jika tidak yakin terhadap suatu ketentuan standar,
  nyatakan bahwa ketentuan tersebut harus diperiksa
  pada dokumen resmi.
- Jangan mengklaim telah membuka atau memeriksa dokumen
  SNI, ACI, ASTM, gambar, atau data yang tidak diberikan.
- Untuk desain atau evaluasi struktur nyata, jelaskan
  batasan analisis dan perlunya pemeriksaan oleh tenaga
  ahli yang kompeten sebelum keputusan pelaksanaan.
- Jika pertanyaan tidak berkaitan dengan Teknik Sipil,
  jelaskan bahwa fokus asisten ini adalah Teknik Sipil.
- Gunakan tabel atau daftar langkah jika membantu.
- Jangan menyatakan jawaban sebagai terverifikasi jika
  belum ada bukti atau data yang memadai.
"""

                    jawaban_ai = jawab_lokal(pertanyaan_ai.strip())
                    if not jawaban_ai:
                        raise ValueError(
                            "Jawaban lokal tidak tersedia."
                       
                        )
                    
                # Simpan percakapan setelah respons berhasil
                st.session_state.riwayat_tanya_teknik_sipil.extend(
                    [
                        {
                            "role": "user",
                            "content": pertanyaan_ai.strip(),
                        },
                        {
                            "role": "assistant",
                            "content": jawaban_ai,
                        },
                    ]
                )

                # Tampilkan jawaban pada pengiriman saat ini
                with st.chat_message("user"):
                    st.markdown(pertanyaan_ai.strip())

                with st.chat_message("assistant"):
                    st.markdown(jawaban_ai)

            except Exception as e:
                st.error(
                    "Permintaan AI gagal diproses. Periksa koneksi, "
                    "API key, akses model, saldo atau batas penggunaan "
                    "API, serta pesan kesalahan pada terminal."
                )
                st.caption(f"Detail teknis: {str(e)}")

    # Peringatan teknis
    st.info(
        "Jawaban AI merupakan bantuan analisis awal. "
        "Verifikasi perhitungan, data lapangan, dokumen resmi, "
        "edisi standar, dan pasal yang relevan sebelum digunakan "
        "untuk desain, audit, atau keputusan konstruksi."
    )

    if st.button(
        "🗑️ Hapus Riwayat Percakapan",
        key="hapus_riwayat_teknik_sipil",
    ):
        st.session_state.riwayat_tanya_teknik_sipil = []
        st.rerun()
