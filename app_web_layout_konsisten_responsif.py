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
st.set_page_config(page_title="AI Structural Monitoring Dashboard", layout="wide", initial_sidebar_state="auto")

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
    
/* Header judul dan logo pendukung */
.header-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 24px;
    width: 100%;
}

.header-text {
    flex: 1;
    min-width: 0;
}

.header-logos {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 14px;
    flex-shrink: 0;
}

.header-logos img {
    width: 100px;
    max-height: 100px;
    object-fit: contain;
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
    [data-testid="stSidebar"], .stSidebar {
        background-color: #0F172A !important;
    }

    /* Tata letak responsif: konten mengikuti lebar layar perangkat. */
    .block-container {
        width: 100%;
        max-width: 100%;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }

    /* Kartu profil khusus HP: tersembunyi pada layar desktop. */
    .st-key-mobile_profile_card {
        display: none;
    }

    @media (max-width: 768px) {
        .st-key-mobile_profile_card {
            display: block !important;
            background: rgba(255,255,255,0.92);
            border: 1px solid #CBD5E1;
            border-radius: 14px;
            padding: 14px;
            margin-bottom: 18px;
            color: #0F172A;
        }

        [data-testid="stMainBlockContainer"], .block-container {
            width: 100% !important;
            max-width: 100% !important;
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
        }

        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.65rem !important;
        }

        [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            min-width: min(100%, 260px) !important;
            flex: 1 1 100% !important;
        }

        .main-title {
            font-size: clamp(1.45rem, 5.5vw, 2rem) !important;
            line-height: 1.25 !important;
            overflow-wrap: anywhere;
        }

        .sub-title {
            font-size: 1rem !important;
            line-height: 1.45 !important;
            margin-bottom: 1rem !important;
        }

        .header-logos img {
            width: 64px !important;
            max-height: 64px !important;
        }

        [data-testid="stImage"] img {
            max-width: 100%;
            height: auto;
            object-fit: contain;
        }

        [data-testid="stPlotlyChart"], [data-testid="stArrowVegaLiteChart"],
        [data-testid="stVegaLiteChart"], [data-testid="stPyplotChart"] {
            max-width: 100% !important;
            overflow-x: auto !important;
        }
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

    st.markdown("<p style='color:#94A3B8; font-size:14px; margin-top:-10px;'>Analis AI Engineer Structure<br>Mahasiswa S2 Teknik Sipil</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown(
        """
        <div style="color:#FFFFFF; font-size:16px; font-weight:bold;">
            📇 Profil & Kontak
        </div>
        <div style="color:#E2E8F0; line-height:2.0; font-size:14px;">
            👨‍💻 <b>Kurniawan Ilyas</b><br>
            🏗️ Analis AI Engineer Structure<br>
            🎓 Mahasiswa S2 Teknik Sipil — UNISSULA<br><br>
            ✉️ <b>Email:</b>
            <a href="mailto:ccms17202020@gmail.com"
               style="color:#7DD3FC;">ccms17202020@gmail.com</a><br>
            📱 <b>Hotline:</b>
            <a href="https://wa.me/6282187948080"
               style="color:#7DD3FC;">6282187948080</a><br>
            ▶️ <b>YouTube:</b>
            <a href="https://www.youtube.com/@cv.konsultanindonesia4341"
               style="color:#7DD3FC;">CV Konsultan Indonesia</a><br>
            📍 <b>Kota:</b> Kendari<br>
            🌏 <b>Provinsi:</b> Sulawesi Tenggara, Indonesia
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    
    st.markdown(
        """
        <div style="
            background-color:#162844;
            color:#FFFFFF;
            padding:16px;
            border-radius:10px;
            font-size:16px;
            line-height:1.8;
            text-align:center;
            font-style:italic;
            border-left:4px solid #38BDF8;
        ">
            <div style="font-size:22px; margin-bottom:10px;">💡</div>
            <b>Bertahan</b> bukan tentang seberapa berat
            beban engkau genggam, tetapi seberapa lama
            beban engkau genggam. 
            <br><br>
            Dalam Mekanika Statika, Mekanika Rekayasa,<br>
            dan Mekanika Bahan, dikenal konsep<br>
            <b>Structural Durability, Long-Term Performance,<br>
            Strength, Creep, Fatigue, dan Structural Performance.</b>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("<p style='text-align:center; color:#64748B; font-size:12px;'>SHM AI System v2.0</p>", unsafe_allow_html=True)

# Profil ringkas khusus HP. Pada desktop kartu ini disembunyikan melalui CSS;
# foto dan profil desktop tetap berada di sidebar.
with st.container(key="mobile_profile_card"):
    foto_col, profil_col = st.columns([1, 2])
    with foto_col:
        st.image("foto_kurniawan.jpg.jpg", width=110)
    with profil_col:
        st.markdown("**👨‍💻 Kurniawan Ilyas**")
        st.caption("Analis AI Engineer Structure")
        st.caption("Mahasiswa S2 Teknik Sipil — UNISSULA")
        st.markdown(
            "✉️ [Email](mailto:ccms17202020@gmail.com)  ·  "
            "📱 [WhatsApp](https://wa.me/6282187948080)  ·  "
            "▶️ [YouTube](https://www.youtube.com/@cv.konsultanindonesia4341)"
        )

# 3. KONTEN UTAMA: HEADER

col_judul, col_logo = st.columns([5, 2])

with col_judul:
    st.markdown(
        '<p class="main-title">🏗️ Dasbor AI Monitoring Gaya Dalam & Keandalan Struktur</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-title">Sistem Peringatan Dini Keandalan Jembatan Berbasis AI</p>',
        unsafe_allow_html=True
    )





# LOGO ITAKI DAN KINDO - SUPPORTED BY SEJAJAR
import base64
from pathlib import Path
import streamlit as st

logo_itaki_b64 = base64.b64encode(
    Path("logo_itaki.png.png").read_bytes()
).decode("utf-8")

logo_kindo_b64 = base64.b64encode(
    Path("logo_kindo.png.png").read_bytes()
).decode("utf-8")

st.markdown(
    f"""
    <div style="
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 6px;
        width: 100%;
        flex-wrap: nowrap;
        margin: 0;
        padding: 0;
    ">
        <span style="
            font-size: 12px;
            font-weight: 600;
            color: #111827;
            white-space: nowrap;
        ">Supported By</span>
        <img src="data:image/png;base64,{logo_itaki_b64}"
             style="width: 38px; height: 38px; object-fit: contain;">
        <img src="data:image/png;base64,{logo_kindo_b64}"
             style="width: 38px; height: 38px; object-fit: contain;">
    </div>
    """,
    unsafe_allow_html=True,
)








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




# AI TANYA JAWAB TEKNIK SIPIL - SIDEBAR
with st.sidebar.expander("🤖 AI Tanya Jawab Teknik Sipil", expanded=False):
    st.caption("Asisten pengetahuan teknik sipil berbasis aturan lokal.")

    with st.form("form_ai_sidebar"):
        pertanyaan_sidebar = st.text_area(
            "Tuliskan pertanyaan Anda:",
            placeholder="Contoh: Apa itu beton?",
            height=100,
        )
        kirim_sidebar = st.form_submit_button("🔎 Cari Jawaban")

    if kirim_sidebar:
        if pertanyaan_sidebar.strip():
            st.session_state["jawaban_sidebar"] = jawab_lokal(
                pertanyaan_sidebar.strip()
            )
        else:
            st.warning("Masukkan pertanyaan terlebih dahulu.")

    if st.session_state.get("jawaban_sidebar"):
        st.markdown("**Jawaban:**")
        st.write(st.session_state["jawaban_sidebar"])

# ============================================================
# 6. KONTEN FRONT-END SETIAP TAB
# ============================================================


tab_dashboard, tab_edukasi, tab_data, tab_analisis = st.tabs(
    [
        "📊 Dasbor Monitoring",
        "📖 Dokumentasi Edukasi",
        "📁 Basis Data Excel",
        "🏗️ Analisis Gaya Dalam & Keandalan Struktur",
    ]
)


# TAB 1: DASBOR MONITORING
with tab_dashboard:
    st.subheader("Analisis Gaya Dalam dan Keandalan Struktur Jembatan")
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
        width="stretch"
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

    try:
        df_baru = pd.DataFrame([data_baru])
        df_gabungan = pd.concat([df, df_baru], ignore_index=True)

        df_gabungan.to_excel(FILE_EXCEL, index=False)

        df = df_gabungan
        st.session_state["data_input_sensor"] = (
            df_gabungan.to_dict(orient="records")
        )

        st.success("Data sensor berhasil disimpan ke file Excel.")
        st.dataframe(df_gabungan.tail(10), width="stretch")

    except Exception as e:
        st.error(f"Gagal menyimpan data ke Excel: {e}")


# TAB 2: DOKUMENTASI EDUKASI
with tab_edukasi:
    st.subheader("Dokumentasi Edukasi Teknik Struktur")

    st.caption(
        "Materi pemeriksaan struktur jembatan: statika, "
        "gaya dalam, kapasitas elemen, data sensor, dan verifikasi."
    )

    # 1. RUMUS DASAR DAN CONTOH PERHITUNGAN
    with st.expander(
        "1. Rumus Dasar dan Contoh Perhitungan",
        expanded=True
    ):
        st.markdown("""
        **Contoh balok sederhana**

        - Bentang: 6 m
        - Beban titik di tengah bentang: 12 kN
        - Tumpuan A: sendi
        - Tumpuan B: rol
        - Berat sendiri dan beban lain diabaikan.

        Persamaan keseimbangan:
        """)

        st.latex(
            r"\sum F_x=0,\quad \sum F_y=0,\quad \sum M=0"
        )

        st.markdown("**Reaksi tumpuan**")
        st.latex(
            r"R_A=R_B=\frac{P}{2}"
            r"=\frac{12}{2}=6\ \mathrm{kN}"
        )

        st.markdown("**Gaya geser maksimum**")
        st.latex(
            r"|V_{\max}|=\frac{P}{2}=6\ \mathrm{kN}"
        )

        st.markdown("**Momen lentur maksimum**")
        st.latex(
            r"M_{\max}=\frac{PL}{4}"
            r"=\frac{12\times6}{4}"
            r"=18\ \mathrm{kN\cdot m}"
        )

        st.info(
            "Hasil berlaku untuk balok sederhana ideal dengan "
            "satu beban titik di tengah bentang. Ini bukan bukti "
            "keamanan jembatan aktual."
        )

    # 2. REAKSI PERLETAKAN DAN KESEIMBANGAN
    with st.expander(
        "2. Reaksi Perletakan dan Keseimbangan Struktur"
    ):
        st.markdown("""
        Untuk balok sederhana dengan beban merata penuh q
        sepanjang bentang L:
        """)

        st.latex(r"\sum F_y=0")
        st.latex(r"R_A+R_B=qL")

        st.markdown("**Untuk beban merata simetris:**")
        st.latex(r"R_A=R_B=\frac{qL}{2}")

        q = st.number_input(
            "Beban merata q (kN/m)",
            min_value=0.0,
            value=10.0,
            step=1.0,
            key="edukasi_q"
        )

        L = st.number_input(
            "Bentang L (m)",
            min_value=0.1,
            value=6.0,
            step=0.5,
            key="edukasi_L"
        )

        RA = q * L / 2
        RB = q * L / 2

        col1, col2 = st.columns(2)
        col1.metric("Reaksi tumpuan A", f"{RA:.2f} kN")
        col2.metric("Reaksi tumpuan B", f"{RB:.2f} kN")

        st.caption(
            "Asumsi: balok sederhana, beban merata penuh, "
            "tanpa beban lain, dan geometri simetris."
        )

    # 3. DIAGRAM GAYA GESER DAN MOMEN LENTUR
    with st.expander(
        "3. Diagram Gaya Geser dan Momen Lentur"
    ):
        st.markdown("""
        Untuk balok sederhana dengan beban merata q
        sepanjang bentang L, gaya geser dan momen pada
        jarak x dari tumpuan A adalah:
        """)

        st.markdown("**Gaya geser**")
        st.latex(r"V(x)=\frac{qL}{2}-qx")

        st.markdown("**Momen lentur**")
        st.latex(r"M(x)=\frac{qL}{2}x-\frac{qx^2}{2}")

        st.markdown("**Gaya geser maksimum**")
        st.latex(r"|V_{\max}|=\frac{qL}{2}")

        st.markdown("**Momen lentur maksimum**")
        st.latex(r"M_{\max}=\frac{qL^2}{8}")

        st.caption(
            "Berlaku untuk balok sederhana dengan beban merata "
            "penuh. Diagram gaya geser linier dan diagram momen "
            "berbentuk parabola."
        )

    # 4. PEMERIKSAAN KAPASITAS LENTUR DAN GESER
    with st.expander(
        "4. Pemeriksaan Kapasitas Lentur dan Geser"
    ):
        st.markdown(
            "Pemeriksaan harus mengikuti jenis material dan "
            "standar desain yang berlaku."
        )

        st.markdown("**Pemeriksaan lentur**")
        st.latex(r"M_u\leq\phi M_n")

        st.markdown("**Pemeriksaan geser**")
        st.latex(r"V_u\leq\phi V_n")

        st.markdown("""
        Keterangan:
        - M_u: momen lentur akibat kombinasi beban desain.
        - V_u: gaya geser akibat kombinasi beban desain.
        - M_n: kapasitas nominal lentur.
        - V_n: kapasitas nominal geser.
        - φ: faktor reduksi kekuatan sesuai standar terkait.
        """)

        st.warning(
            "Belum dapat ditentukan aman atau tidak aman tanpa "
            "data penampang, material, tulangan atau profil, "
            "panjang efektif, kondisi tumpuan, dan kombinasi beban."
        )

    # 5. INTERPRETASI RESPONS SENSOR
    with st.expander(
        "5. Interpretasi Respons Sensor dan Keterbatasan Data"
    ):
        st.markdown("""
        Bedakan jenis data berikut:

        1. Data aktual dari sensor dan sistem akuisisi.
        2. Data simulasi atau input manual.
        3. Data hasil pengolahan atau estimasi algoritma.

        Nilai microstrain menunjukkan regangan, bukan langsung
        tegangan, kapasitas, atau tingkat keamanan struktur.
        """)

        st.markdown("**Hubungan tegangan dan regangan elastis**")
        st.latex(r"\sigma=E\varepsilon")

        st.markdown(
            "Konversi microstrain menjadi regangan tak berdimensi:"
        )
        st.latex(
            r"\varepsilon=\varepsilon_{\mu\varepsilon}\times10^{-6}"
        )

        st.markdown("""
        Keterangan:
        - σ: tegangan normal.
        - E: modulus elastisitas material.
        - ε: regangan tak berdimensi.
        """)

        st.warning(
            "Interpretasi memerlukan informasi kalibrasi sensor, "
            "lokasi pemasangan, satuan, waktu pengukuran, kualitas "
            "sinyal, dan pengaruh lingkungan. Hubungan σ = Eε "
            "berlaku dengan asumsi elastis linear yang sesuai."
        )

    # 6. VERIFIKASI GAMBAR, SPESIFIKASI, DAN STANDAR
    with st.expander(
        "6. Verifikasi Gambar, Spesifikasi, dan Standar Desain"
    ):
        st.markdown("""
        **Daftar periksa verifikasi:**

        1. Periksa identitas, revisi, dan tanggal gambar.
        2. Cocokkan dimensi lapangan dengan gambar desain.
        3. Verifikasi mutu material berdasarkan dokumen dan hasil uji.
        4. Periksa beban dan kombinasi beban yang digunakan.
        5. Bandingkan gaya dalam dengan kapasitas elemen.
        6. Catat penyimpangan, bukti, dan tindak lanjut.
        """)

        status = st.selectbox(
            "Status pemeriksaan edukatif",
            [
                "Belum diperiksa",
                "Sesuai berdasarkan bukti yang tersedia",
                "Ada penyimpangan",
                "Data belum mencukupi"
            ],
            key="edukasi_status_verifikasi"
        )

        catatan = st.text_area(
            "Catatan dan bukti pemeriksaan",
            placeholder=(
                "Masukkan nomor gambar, lokasi elemen, "
                "hasil pengukuran, atau temuan."
            ),
            key="edukasi_catatan_verifikasi"
        )

        st.write("Status yang dipilih:", status)

        if catatan.strip():
            st.write("Catatan:", catatan)

        st.caption(
            "Status dan catatan ini hanya untuk demonstrasi. "
            "Belum tersimpan sebagai rekaman audit permanen."
        )

        st.markdown("**Rujukan standar yang perlu diverifikasi**")
        st.markdown("""
        - SNI 1725:2016 — Pembebanan untuk Jembatan.
        - SNI 2833:2016 — Perancangan Jembatan terhadap Beban Gempa.
        """)

        st.warning(
            "Periksa status keberlakuan, standar pengganti, "
            "dan nomor pasal pada dokumen resmi BSN atau "
            "Bina Marga sebelum menjadikannya dasar keputusan teknis."
        )


# TAB 3: BASIS DATA EXCEL
with tab_data:
    st.subheader("Basis Data Sensor Jembatan")
    st.markdown("### Input Data Sensor")


    st.dataframe(df, width="stretch")

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
    st.dataframe(df, width="stretch")

    st.info(
        "Data sensor ditampilkan untuk pemeriksaan awal. "
        "Penilaian keandalan struktur memerlukan verifikasi "
        "model, beban, kapasitas, dan validitas data sensor."
    )

    
    st.subheader("Skema Elemen Struktur Jembatan")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 3))

    # Gelagar utama: skema ilustratif, bukan gambar as-built
    ax.plot([0, 10], [1, 1], linewidth=8, color="steelblue")

    # Tumpuan kiri dan kanan
    ax.plot([0, 0.5, -0.5, 0], [1, 0, 0, 1], color="black")
    ax.plot([10, 10.5, 9.5, 10], [1, 0, 0, 1], color="black")

    # Titik sensor
    posisi_sensor = [0, 5, 10]
    label_sensor = [
        "Tumpuan Kiri",
        "Tengah Bentang",
        "Tumpuan Kanan",
    ]

    for x, label in zip(posisi_sensor, label_sensor):
        ax.scatter(x, 1, color="red", s=90, zorder=3)
        ax.annotate(
            label,
            (x, 1),
            xytext=(0, 18),
            textcoords="offset points",
            ha="center",
        )

    ax.set_xlim(-1, 11)
    ax.set_ylim(-0.5, 2)
    ax.set_xlabel("Posisi sepanjang bentang (skema)")
    ax.set_yticks([])
    ax.set_title("Skema Gelagar dan Lokasi Sensor")
    ax.grid(axis="x", alpha=0.25)

    st.pyplot(fig)
    plt.close(fig)
    
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
