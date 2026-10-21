# ============================================================
# 6. TABS DASHBOARD
# ============================================================
tab_dashboard, tab_edukasi, tab_data = st.tabs(
    ["📊 Dasbor Monitoring", "📖 Dokumentasi Edukasi", "💾 Basis Data Excel"]
)

with tab_dashboard:
    st.warning(
        "Mode saat ini menggunakan input slider untuk simulasi, bukan pembacaan "
        "sensor IoT real-time. Status di bawah bukan keputusan keselamatan struktur."
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("R² pada data uji", f"{r2:.3f}" if np.isfinite(r2) else "Tidak tersedia")
    with m2:
        st.metric("RMSE data uji (kNm)", f"{rmse:.3f}")
    with m3:
        st.metric("Baris dataset valid", f"{len(df)}")

    st.caption(
        "R² dapat bernilai negatif. Jangan menyebut R² × 100 sebagai persentase "
        "akurasi tanpa definisi evaluasi yang sesuai."
    )

    left_col, right_col = st.columns([1.1, 0.9])

    with left_col:
        st.markdown("#### 🔮 Panel Input Sensor")
        st.caption("Ubah nilai berikut untuk menjalankan simulasi prediksi.")

        input_kiri = st.slider(
            "Sensor Tumpuan Kiri (microstrain)", 0.0, 1000.0, 250.0, 1.0
        )
        input_tengah = st.slider(
            "Sensor Tengah Bentang (microstrain)", 0.0, 1000.0, 400.0, 1.0
        )
        input_kanan = st.slider(
            "Sensor Tumpuan Kanan (microstrain)", 0.0, 1000.0, 150.0, 1.0
        )

        data_sensor_baru = pd.DataFrame(
            [[input_kiri, input_tengah, input_kanan]],
            columns=FITUR,
        )
        data_sensor_scaled = scaler.transform(data_sensor_baru)
        prediksi_momen = float(ai_model.predict(data_sensor_scaled)[0])

        st.metric("Estimasi momen hasil AI (kNm)", f"{prediksi_momen:.2f}")

        # Ambang ini hanya contoh konfigurasi aplikasi.
        # Jangan menafsirkannya sebagai batas layan/ultimit tanpa verifikasi desain.
        st.markdown("#### Indikator berbasis ambang sementara")
        st.caption(
            "Ambang 300 dan 500 kNm di bawah hanya contoh pemrograman. "
            "Ganti berdasarkan kriteria teknis yang disahkan untuk struktur yang dinilai."
        )

        if prediksi_momen > 500:
            st.error("🚨 INDIKATOR: MELEBIHI AMBANG ATAS SEMENTARA")
        elif prediksi_momen > 300:
            st.warning("⚠️ INDIKATOR: MELEBIHI AMBANG BAWAH SEMENTARA")
        else:
            st.info("ℹ️ INDIKATOR: DI BAWAH AMBANG BAWAH SEMENTARA")

        st.download_button(
            label="📥 Unduh Laporan Simulasi (.txt)",
            data=(
                "LAPORAN SIMULASI PREDIKSI AI\n"
                "Sistem: AI Structural Monitoring Dashboard\n"
                "Mode: Input slider/simulasi, bukan pembacaan sensor langsung\n\n"
                f"Sensor kiri: {input_kiri:.2f} microstrain\n"
                f"Sensor tengah: {input_tengah:.2f} microstrain\n"
                f"Sensor kanan: {input_kanan:.2f} microstrain\n"
                f"Estimasi momen AI: {prediksi_momen:.2f} kNm\n"
                f"R2 data uji: {r2:.4f}\n"
                f"MSE data uji: {mse:.4f}\n"
                f"RMSE data uji: {rmse:.4f} kNm\n\n"
                "PERINGATAN: Hasil ini merupakan estimasi model dan bukan "
                "sertifikasi keselamatan struktur. Verifikasi dengan analisis "
                "teknik dan data lapangan yang independen.\n"
            ),
            file_name="Laporan_Simulasi_Monitoring_AI.txt",
            mime="text/plain",
        )

    with right_col:
        st.markdown("#### 📉 Grafik Prediksi terhadap Target Data Uji")
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        ax.scatter(y_test, predictions, alpha=0.65, label="Prediksi MLP")

        nilai_min = min(float(y_test.min()), float(np.min(predictions)))
        nilai_max = max(float(y_test.max()), float(np.max(predictions)))
        ax.plot(
            [nilai_min, nilai_max],
            [nilai_min, nilai_max],
            linestyle="--",
            linewidth=2,
            label="Garis identitas",
        )
        ax.set_xlabel("Target pada dataset uji (kNm)")
        ax.set_ylabel("Prediksi model (kNm)")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

with tab_edukasi:
    st.markdown("### 📘 Dasar Teori Structural Health Monitoring")
    st.write(
        """
        Model MLP mempelajari pola statistik antara fitur microstrain dan
        target momen yang tersedia pada dataset. Model tidak otomatis
        menghitung gaya dalam berdasarkan mekanika struktur; ia mengestimasi
        target berdasarkan pola data latih.

        Agar dapat digunakan sebagai sistem pendukung pemantauan, perlu dilakukan
        validasi terhadap data independen, pemeriksaan kalibrasi sensor, evaluasi
        rentang input, analisis ketidakpastian, serta verifikasi rekayasa struktur.
        """
    )

with tab_data:
    st.markdown("#### 💾 Dataset yang digunakan")
    st.caption(
