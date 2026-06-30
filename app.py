import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Prediksi Gaya Belajar & IPK Mahasiswa", page_icon="\U0001F393", layout="wide")

# ============================================================
# KONFIGURASI FITUR (hasil ekstraksi otomatis dari notebook)
# ============================================================

# --- Fitur Notebook 1: Klasifikasi Gaya Belajar (20 kolom, urutan HARUS sama persis) ---
CLF_COLUMNS = [
    " Kualitas akses internet untuk belajar  ",
    " Kehadiran perkuliahan saya selama satu semester terakhir secara umum  ",
    " Saya lebih mudah memahami materi jika disajikan dalam bentuk diagram, bagan, atau infografik.  ",
    "Saya lebih mudah mengingat materi ketika dosen menggunakan slide yang jelas dan visual. ",
    "Warna, simbol, atau penanda visual membantu saya memahami konsep. ",
    "Saya lebih suka membuat peta konsep atau mind map saat belajar. ",
    "Saya lebih cepat memahami penjelasan jika disertai ilustrasi atau video. ",
    "Saya lebih mudah memahami materi dengan mendengarkan penjelasan lisan dosen. ",
    "Diskusi kelas atau diskusi kelompok membantu saya belajar lebih baik. ",
    "Saya sering mengulang materi dengan membacanya keras-keras atau menjelaskannya secara verbal. ",
    "Saya lebih mudah menangkap konsep ketika mendengar contoh yang dijelaskan secara lisan. ",
    "Saya merasa rekaman audio atau penjelasan verbal membantu saya belajar. ",
    "Saya lebih suka belajar dari buku, modul, atau artikel tertulis. ",
    "Saya lebih mudah memahami materi dengan mencatat ulang poin-poin penting. ",
    "Saya suka membuat rangkuman tertulis setelah perkuliahan. ",
    "Saya lebih nyaman menerima instruksi dalam bentuk tulisan daripada penjelasan lisan. ",
    "Saya lebih mudah menghafal istilah atau konsep ketika menuliskannya. ",
    "Saya lebih mudah memahami materi ketika langsung mempraktikkannya. ",
    "Contoh kasus nyata membuat saya lebih cepat memahami konsep. ",
    "Saya belajar lebih baik melalui simulasi, eksperimen, atau demonstrasi. "
]

CLF_FRIENDLY_LABELS = [
    "Kualitas akses internet untuk belajar",
    "Kehadiran perkuliahan semester terakhir",
    "Mudah paham lewat diagram/bagan/infografik",
    "Mudah ingat lewat slide visual dosen",
    "Warna/simbol/penanda visual membantu paham",
    "Suka membuat peta konsep / mind map",
    "Cepat paham jika disertai ilustrasi/video",
    "Mudah paham dengan mendengar penjelasan lisan dosen",
    "Diskusi kelas/kelompok membantu belajar",
    "Mengulang materi dengan membaca keras-keras / verbal",
    "Mudah tangkap konsep dari contoh lisan",
    "Rekaman audio/penjelasan verbal membantu belajar",
    "Suka belajar dari buku/modul/artikel tertulis",
    "Mudah paham dengan mencatat ulang poin penting",
    "Suka membuat rangkuman tertulis setelah kuliah",
    "Lebih nyaman instruksi tertulis daripada lisan",
    "Mudah menghafal dengan menuliskannya",
    "Mudah paham dengan langsung praktik",
    "Contoh kasus nyata mempercepat paham",
    "Belajar lebih baik lewat simulasi/eksperimen"
]

LE_CLASSES = [
    "Auditory",
    "Kinesthetic",
    "ReadWrite",
    "Visual"
]  # urutan label hasil LabelEncoder (alfabetis)

# --- Fitur Notebook 2: Regresi Prediksi IPK ---
FITUR_NUMERIK = [
    " Usia  (tahun)",
    " Kualitas akses internet untuk belajar  ",
    " Kehadiran perkuliahan saya selama satu semester terakhir secara umum  ",
    " Saya lebih mudah memahami materi jika disajikan dalam bentuk diagram, bagan, atau infografik.  ",
    "Warna, simbol, atau penanda visual membantu saya memahami konsep. ",
    "Saya lebih suka membuat peta konsep atau mind map saat belajar. ",
    "Saya lebih mudah memahami materi dengan mendengarkan penjelasan lisan dosen. ",
    "Diskusi kelas atau diskusi kelompok membantu saya belajar lebih baik. ",
    "Saya sering mengulang materi dengan membacanya keras-keras atau menjelaskannya secara verbal. ",
    "Saya lebih mudah menangkap konsep ketika mendengar contoh yang dijelaskan secara lisan. ",
    "Saya merasa rekaman audio atau penjelasan verbal membantu saya belajar. ",
    "Saya lebih suka belajar dari buku, modul, atau artikel tertulis. ",
    "Saya lebih mudah memahami materi dengan mencatat ulang poin-poin penting. ",
    "Saya suka membuat rangkuman tertulis setelah perkuliahan. ",
    "Saya lebih nyaman menerima instruksi dalam bentuk tulisan daripada penjelasan lisan. ",
    "Saya lebih mudah menghafal istilah atau konsep ketika menuliskannya. ",
    "Saya lebih mudah memahami materi ketika langsung mempraktikkannya. ",
    "Contoh kasus nyata membuat saya lebih cepat memahami konsep. ",
    "Saya belajar lebih baik melalui simulasi, eksperimen, atau demonstrasi. ",
    "Saya cenderung memahami materi setelah mencoba sendiri langkah-langkahnya. ",
    "Saya lebih suka tugas yang melibatkan praktik daripada hanya membaca teori. ",
    " Saya memiliki jadwal belajar yang cukup teratur setiap minggu.  ",
    "Saya mampu menentukan prioritas antara kuliah, tugas, dan kegiatan lain. ",
    "Saya meninjau kembali materi kuliah setelah pertemuan selesai. ",
    "Saya mencari sumber tambahan ketika ada materi yang belum saya pahami. ",
    "Saya membandingkan berbagai sumber untuk memahami suatu topik. ",
    "Saya membuat ringkasan, poin penting, atau catatan pribadi dari materi kuliah. ",
    "Saya mengecek sendiri apakah saya benar-benar sudah memahami materi. ",
    "Saya memiliki target akademik yang jelas setiap semester. ",
    "Saya tetap berusaha memahami materi walaupun terasa sulit. ",
    "Saya terdorong untuk belajar bukan hanya demi nilai, tetapi juga demi pemahaman. ",
    "Saya merasa bertanggung jawab terhadap hasil belajar saya sendiri. ",
    "Saya berusaha memperbaiki strategi belajar ketika hasil saya kurang baik. ",
    "Saya aktif bertanya atau berdiskusi ketika ada materi yang belum jelas. ",
    "Saya mengerjakan tugas kuliah dengan sungguh-sungguh. ",
    "Saya mengikuti perkuliahan dengan fokus. ",
    "Saya memanfaatkan umpan balik dari dosen untuk memperbaiki hasil belajar saya. ",
    "Saya memiliki tempat belajar yang cukup nyaman. ",
    "Kondisi lingkungan tempat saya belajar mendukung konsentrasi saya. ",
    "Saya memiliki akses perangkat yang memadai untuk mengerjakan tugas kuliah. ",
    "Saya memiliki dukungan sosial yang cukup dari keluarga atau teman untuk belajar. ",
    "Hambatan teknis seperti internet atau perangkat sering mengganggu proses belajar saya. ",
    "Saya memanfaatkan platform digital untuk memahami materi kuliah. ",
    "Saya menggunakan aplikasi atau tools digital untuk mencatat, merangkum, atau mengatur belajar. ",
    "Saya dapat belajar secara efektif melalui media pembelajaran online. ",
    "Saya mudah terdistraksi oleh penggunaan perangkat digital saat sedang belajar. ",
    "Saat membaca materi, saya memberi highlight, warna, simbol, atau tanda khusus pada bagian penting. ",
    "Saat belajar di rumah, saya membuat mind map, peta konsep, atau skema ringkas materi. ",
    "Saat mengulang pelajaran, saya menggunakan gambar, ilustrasi, tabel, atau tampilan visual lain sebagai bantuan belajar. ",
    "Saat belajar di rumah, saya mendengarkan ulang rekaman penjelasan atau audio pembelajaran. ",
    "Saat belajar sendiri, saya membaca materi dengan suara keras agar lebih mudah memahami isi materi. ",
    "Saat mengulang pelajaran, saya menjelaskan materi dengan lisan kepada diri sendiri seolah-olah sedang mengajar orang lain. ",
    "Saat belajar di rumah, saya berdiskusi dengan teman, baik secara langsung maupun online, untuk memahami materi. ",
    "Saat belajar di rumah, saya membaca ulang catatan kuliah, modul, buku, atau artikel untuk memahami materi. ",
    "Saat belajar sendiri, saya menulis ulang poin-poin penting dari materi yang sedang dipelajari. ",
    "Setelah belajar, saya membuat rangkuman tertulis dengan bahasa saya sendiri. ",
    "Saat menemukan materi yang sulit, saya mencari penjelasan tertulis dari buku, modul, artikel, atau internet. ",
    "Saat menghafal konsep atau istilah, saya menuliskannya berulang kali agar lebih ingat. ",
    "Saat belajar di rumah, saya mengerjakan latihan soal, tugas, atau kuis untuk memahami materi. ",
    "Saat belajar sendiri, saya mencoba langsung langkah-langkah atau prosedur yang sedang dipelajari. ",
    "Saat mempelajari materi, saya menggunakan contoh kasus nyata agar lebih mudah memahami konsep. ",
    "Saat belajar di rumah, saya melakukan simulasi, praktik, eksperimen, atau demonstrasi kecil sesuai materi yang dipelajari. ",
    "Saat merasa belum paham, saya mencoba sendiri sampai mengetahui cara kerja atau penyelesaian dari materi tersebut. ",
    "Saat merasa belum paham, saya mencoba sendiri sampai mengetahui cara kerja atau penyelesaian dari materi tersebut.  2",
    " IPS semester terakhir  "
]

FITUR_KATEGORIKAL = [
    " Program studi  ",
    " Semester saat ini  ",
    " Jenis kelamin  ",
    " Status tempat tinggal selama kuliah  ",
    " Apakah Anda bekerja sambil kuliah?  ",
    " Sumber pembiayaan utama kuliah saya saat ini adalah:  ",
    " Perangkat utama yang digunakan untuk belajar  ",
    "Dalam satu semester terakhir, apakah Anda pernah mengulang mata kuliah?  ",
    "Dalam satu semester terakhir, apakah Anda pernah mendapat nilai D/E? "
]

KATEGORIKAL_VALUES = {
    " Program studi  ": ["241127895", "Agribisnis ", "Agrotek", "Akuntansi", "Akuntansi ", "Akutansi ", "Arsitektur", "Bisnis Digital ", "D3 Analis Kesehatan", "D3 Gizi", "D4 Analis Kesehatan", "Desain Interior", "Ekonomi Pembangunan ", "Farmasi", "Farmasj", "Fisika", "Fk", "Gizi", "Hukum", "INFORMATIKA", "IUPIE", "Ilmu Hukum", "Ilmu Keperawatan", "Ilmu Komputer", "Ilmu Komunikasi", "Ilmu Komunikasi & Multimedia", "Ilmu Komunikasi Internasional", "Ilmu Komunikasi dan Multimedia", "Informatika", "Informatika ", "Kedokteran ", "Kedokteran gigi", "Kesehatan Masyarakat ", "Komunikasi", "MANAJEMEN S1", "Managemen", "Management", "Management ", "Manajamen", "Manajemen", "Manajemen ", "Manajemen Haji dan Umrah", "Manajemen Hutan", "Manajemen Keuangan Negara", "Manajemen S1", "Manajemen Sumberdaya Perairan", "Manajemen bisnis", "Matematika", "PGMI", "PGSD", "Pendidikan Agama Kristen", "Pendidikan Dokter", "Pendidikan Keagamaan Katolik ", "Pendidikan Matematika ", "Pendidikan Teknik Informatika dan Komputer", "Perbankan Syariah", "Perencanaan Tata Ruang dan Pertahanan ", "Perpajakan", "Rekayasa Perancangan Mekanik ", "S1 AGRIBISNIS ", "S1 AKUNTANSI", "S1 Agribisnis", "S1 Agribisnis ", "S1 Akuntansi", "S1 Akuntansi ", "S1 Farmasi", "S1 Farmasi ", "S1 Ilmu Komputer", "S1 Ilmu Komputer ", "S1 Informatika ", "S1 Manajemen", "S1 Teknik Geodesi", "S1 akuntansi", "S1 farmasi", "S1 farmasi ", "S1 manajemen ", "S1-FARMASI ", "S1-Keperawatan", "S2 Manajemen", "Sains Data", "Sastra Inggris", "Sastra informatika", "Seni Rupa ", "Sistem Informasi", "Sistem Informasi ", "Sistem informasi", "Sistem informasi ", "Sosiologi", "TI", "Teknik", "Teknik Biomedis ", "Teknik Elektro", "Teknik Industr", "Teknik Industri", "Teknik Industri ", "Teknik Indystri", "Teknik Informatika", "Teknik Pengelolaan dan Perawatan Alat Berat", "Teknik Sipil", "Teknik Sipil ", "Teknik Sistem Energi ", "Teknik industri", "Teknik industri ", "Teknik sipil", "Teknologi Pangan", "Teknologi pangan ", "Terapi Wicara", "arsitektur", "fkg", "hukum", "infomatika", "informatika", "komunikasi", "manajemen", "manajemen ", "psikologi", "s1 farmasi", "sistem informas ", "sistem informasi", "teknik informatika"],
    " Semester saat ini  ": ["1", "2", "3", "4", "5", "6", "7", "8 atau lebih"],
    " Jenis kelamin  ": ["Laki-laki", "Perempuan"],
    " Status tempat tinggal selama kuliah  ": ["Bersama nenek", "Bersama orang tua", "Kontrak/tinggal mandiri", "Kost/asrama", "Wali"],
    " Apakah Anda bekerja sambil kuliah?  ": ["Tidak", "Ya, paruh waktu", "Ya, penuh waktu"],
    " Sumber pembiayaan utama kuliah saya saat ini adalah:  ": ["Beasiswa penuh", "Beasiswa penuh, Bekerja sambil kuliah", "Beasiswa sebagian", "Bekerja sambil kuliah", "Biaya sendiri / mandiri", "Biaya sendiri / mandiri, Bekerja sambil kuliah", "Orang tua / keluarga", "Orang tua / keluarga, Beasiswa penuh", "Orang tua / keluarga, Beasiswa penuh, Biaya sendiri / mandiri, Bekerja sambil kuliah", "Orang tua / keluarga, Beasiswa sebagian", "Orang tua / keluarga, Beasiswa sebagian, Biaya sendiri / mandiri", "Orang tua / keluarga, Beasiswa sebagian, Biaya sendiri / mandiri, Bekerja sambil kuliah", "Orang tua / keluarga, Bekerja sambil kuliah", "Orang tua / keluarga, Biaya sendiri / mandiri", "Orang tua / keluarga, Biaya sendiri / mandiri, Bekerja sambil kuliah"],
    " Perangkat utama yang digunakan untuk belajar  ": ["Komputer desktop", "Laptop", "Laptop, Komputer desktop", "Laptop, Smartphone", "Laptop, Smartphone, Komputer desktop", "Laptop, Smartphone, Tablet", "Laptop, Smartphone, Tablet, Komputer desktop", "Laptop, Tablet", "Smartphone", "Smartphone, Tablet", "Tablet"],
    "Dalam satu semester terakhir, apakah Anda pernah mengulang mata kuliah?  ": ["Tidak", "Ya"],
    "Dalam satu semester terakhir, apakah Anda pernah mendapat nilai D/E? ": ["Tidak", "Ya"]
}

X_COLUMNS_ORDER = [
    " Program studi  ",
    " Semester saat ini  ",
    " Jenis kelamin  ",
    " Usia  (tahun)",
    " Status tempat tinggal selama kuliah  ",
    " Apakah Anda bekerja sambil kuliah?  ",
    " Sumber pembiayaan utama kuliah saya saat ini adalah:  ",
    " Perangkat utama yang digunakan untuk belajar  ",
    " Kualitas akses internet untuk belajar  ",
    " Kehadiran perkuliahan saya selama satu semester terakhir secara umum  ",
    " Saya lebih mudah memahami materi jika disajikan dalam bentuk diagram, bagan, atau infografik.  ",
    "Warna, simbol, atau penanda visual membantu saya memahami konsep. ",
    "Saya lebih suka membuat peta konsep atau mind map saat belajar. ",
    "Saya lebih mudah memahami materi dengan mendengarkan penjelasan lisan dosen. ",
    "Diskusi kelas atau diskusi kelompok membantu saya belajar lebih baik. ",
    "Saya sering mengulang materi dengan membacanya keras-keras atau menjelaskannya secara verbal. ",
    "Saya lebih mudah menangkap konsep ketika mendengar contoh yang dijelaskan secara lisan. ",
    "Saya merasa rekaman audio atau penjelasan verbal membantu saya belajar. ",
    "Saya lebih suka belajar dari buku, modul, atau artikel tertulis. ",
    "Saya lebih mudah memahami materi dengan mencatat ulang poin-poin penting. ",
    "Saya suka membuat rangkuman tertulis setelah perkuliahan. ",
    "Saya lebih nyaman menerima instruksi dalam bentuk tulisan daripada penjelasan lisan. ",
    "Saya lebih mudah menghafal istilah atau konsep ketika menuliskannya. ",
    "Saya lebih mudah memahami materi ketika langsung mempraktikkannya. ",
    "Contoh kasus nyata membuat saya lebih cepat memahami konsep. ",
    "Saya belajar lebih baik melalui simulasi, eksperimen, atau demonstrasi. ",
    "Saya cenderung memahami materi setelah mencoba sendiri langkah-langkahnya. ",
    "Saya lebih suka tugas yang melibatkan praktik daripada hanya membaca teori. ",
    " Saya memiliki jadwal belajar yang cukup teratur setiap minggu.  ",
    "Saya mampu menentukan prioritas antara kuliah, tugas, dan kegiatan lain. ",
    "Saya meninjau kembali materi kuliah setelah pertemuan selesai. ",
    "Saya mencari sumber tambahan ketika ada materi yang belum saya pahami. ",
    "Saya membandingkan berbagai sumber untuk memahami suatu topik. ",
    "Saya membuat ringkasan, poin penting, atau catatan pribadi dari materi kuliah. ",
    "Saya mengecek sendiri apakah saya benar-benar sudah memahami materi. ",
    "Saya memiliki target akademik yang jelas setiap semester. ",
    "Saya tetap berusaha memahami materi walaupun terasa sulit. ",
    "Saya terdorong untuk belajar bukan hanya demi nilai, tetapi juga demi pemahaman. ",
    "Saya merasa bertanggung jawab terhadap hasil belajar saya sendiri. ",
    "Saya berusaha memperbaiki strategi belajar ketika hasil saya kurang baik. ",
    "Saya aktif bertanya atau berdiskusi ketika ada materi yang belum jelas. ",
    "Saya mengerjakan tugas kuliah dengan sungguh-sungguh. ",
    "Saya mengikuti perkuliahan dengan fokus. ",
    "Saya memanfaatkan umpan balik dari dosen untuk memperbaiki hasil belajar saya. ",
    "Saya memiliki tempat belajar yang cukup nyaman. ",
    "Kondisi lingkungan tempat saya belajar mendukung konsentrasi saya. ",
    "Saya memiliki akses perangkat yang memadai untuk mengerjakan tugas kuliah. ",
    "Saya memiliki dukungan sosial yang cukup dari keluarga atau teman untuk belajar. ",
    "Hambatan teknis seperti internet atau perangkat sering mengganggu proses belajar saya. ",
    "Saya memanfaatkan platform digital untuk memahami materi kuliah. ",
    "Saya menggunakan aplikasi atau tools digital untuk mencatat, merangkum, atau mengatur belajar. ",
    "Saya dapat belajar secara efektif melalui media pembelajaran online. ",
    "Saya mudah terdistraksi oleh penggunaan perangkat digital saat sedang belajar. ",
    "Saat membaca materi, saya memberi highlight, warna, simbol, atau tanda khusus pada bagian penting. ",
    "Saat belajar di rumah, saya membuat mind map, peta konsep, atau skema ringkas materi. ",
    "Saat mengulang pelajaran, saya menggunakan gambar, ilustrasi, tabel, atau tampilan visual lain sebagai bantuan belajar. ",
    "Saat belajar di rumah, saya mendengarkan ulang rekaman penjelasan atau audio pembelajaran. ",
    "Saat belajar sendiri, saya membaca materi dengan suara keras agar lebih mudah memahami isi materi. ",
    "Saat mengulang pelajaran, saya menjelaskan materi dengan lisan kepada diri sendiri seolah-olah sedang mengajar orang lain. ",
    "Saat belajar di rumah, saya berdiskusi dengan teman, baik secara langsung maupun online, untuk memahami materi. ",
    "Saat belajar di rumah, saya membaca ulang catatan kuliah, modul, buku, atau artikel untuk memahami materi. ",
    "Saat belajar sendiri, saya menulis ulang poin-poin penting dari materi yang sedang dipelajari. ",
    "Setelah belajar, saya membuat rangkuman tertulis dengan bahasa saya sendiri. ",
    "Saat menemukan materi yang sulit, saya mencari penjelasan tertulis dari buku, modul, artikel, atau internet. ",
    "Saat menghafal konsep atau istilah, saya menuliskannya berulang kali agar lebih ingat. ",
    "Saat belajar di rumah, saya mengerjakan latihan soal, tugas, atau kuis untuk memahami materi. ",
    "Saat belajar sendiri, saya mencoba langsung langkah-langkah atau prosedur yang sedang dipelajari. ",
    "Saat mempelajari materi, saya menggunakan contoh kasus nyata agar lebih mudah memahami konsep. ",
    "Saat belajar di rumah, saya melakukan simulasi, praktik, eksperimen, atau demonstrasi kecil sesuai materi yang dipelajari. ",
    "Saat merasa belum paham, saya mencoba sendiri sampai mengetahui cara kerja atau penyelesaian dari materi tersebut. ",
    "Saat merasa belum paham, saya mencoba sendiri sampai mengetahui cara kerja atau penyelesaian dari materi tersebut.  2",
    " IPS semester terakhir  ",
    "Dalam satu semester terakhir, apakah Anda pernah mengulang mata kuliah?  ",
    "Dalam satu semester terakhir, apakah Anda pernah mendapat nilai D/E? "
]

# Kolom Likert (skala 1-5) di antara fitur numerik regresi
LIKERT_EXCLUDE = [" Usia  (tahun)", " IPS semester terakhir  "]


# ============================================================
# LOAD MODEL (cache supaya tidak reload setiap interaksi)
# ============================================================
@st.cache_resource
def load_models():
    with open("BestModel_Klasifikasi_RandomForest_Kelompok3.pkl", "rb") as f:
        model_klasifikasi = pickle.load(f)
    with open("BestModel_Regresi_OptimizedRandomForest_Kelompok3.pkl", "rb") as f:
        model_regresi = pickle.load(f)
    return model_klasifikasi, model_regresi

try:
    model_klasifikasi, model_regresi = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error = str(e)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("\U0001F393 Menu Aplikasi")
menu = st.sidebar.radio(
    "Pilih Menu:",
    ["\U0001F3A8 Klasifikasi Gaya Belajar", "\U0001F4CA Prediksi IPK"]
)
st.sidebar.markdown("---")
st.sidebar.caption("UAS Pembelajaran Mesin - Kelompok 3")
st.sidebar.caption("Model: Random Forest (Klasifikasi) & Optimized Random Forest (Regresi)")

if not models_loaded:
    st.error(f"Gagal memuat model. Pastikan file .pkl berada di folder yang sama dengan app.py.\n\nDetail error: {load_error}")
    st.stop()


# ============================================================
# MENU 1: KLASIFIKASI GAYA BELAJAR
# ============================================================
if menu == "\U0001F3A8 Klasifikasi Gaya Belajar":
    st.title("\U0001F3A8 Klasifikasi Gaya Belajar Mahasiswa")
    st.markdown(
        "Isi jawaban kuisioner gaya belajar di bawah ini (skala **1 = Sangat Tidak Setuju** "
        "sampai **5 = Sangat Setuju**), lalu klik **Prediksi** untuk mengetahui gaya belajar dominan kamu."
    )
    st.markdown("---")

    with st.form("form_klasifikasi"):
        jawaban = []

        st.subheader("\U0001F4F6 Akses & Kehadiran")
        v0 = st.slider(CLF_FRIENDLY_LABELS[0], 1, 5, 3)
        v1 = st.slider(CLF_FRIENDLY_LABELS[1], 1, 5, 3)
        jawaban.extend([v0, v1])

        st.subheader("\U0001F441\ufe0f Dimensi Visual")
        for i in range(2, 7):
            jawaban.append(st.slider(CLF_FRIENDLY_LABELS[i], 1, 5, 3, key=f"clf_{i}"))

        st.subheader("\U0001F442 Dimensi Auditory")
        for i in range(7, 12):
            jawaban.append(st.slider(CLF_FRIENDLY_LABELS[i], 1, 5, 3, key=f"clf_{i}"))

        st.subheader("\U0001F4DD Dimensi Read/Write")
        for i in range(12, 17):
            jawaban.append(st.slider(CLF_FRIENDLY_LABELS[i], 1, 5, 3, key=f"clf_{i}"))

        st.subheader("\U0001F3C3 Dimensi Kinesthetic")
        for i in range(17, 20):
            jawaban.append(st.slider(CLF_FRIENDLY_LABELS[i], 1, 5, 3, key=f"clf_{i}"))

        submitted = st.form_submit_button("\U0001F50D Prediksi Gaya Belajar", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([jawaban], columns=CLF_COLUMNS)
        try:
            pred_encoded = model_klasifikasi.predict(input_df)[0]
            # Model imblearn Pipeline mengeluarkan label terenkode (0-3) sesuai urutan LabelEncoder
            if isinstance(pred_encoded, (int, np.integer)):
                pred_label = LE_CLASSES[pred_encoded]
            else:
                pred_label = str(pred_encoded)

            st.markdown("---")
            st.success(f"### \U0001F3AF Gaya Belajar Dominan Kamu: **{pred_label}**")

            label_desc = {
                "Visual": "Kamu lebih mudah memahami materi lewat diagram, gambar, video, dan visualisasi.",
                "Auditory": "Kamu lebih mudah memahami materi lewat penjelasan lisan, diskusi, dan audio.",
                "ReadWrite": "Kamu lebih mudah memahami materi lewat membaca dan menulis catatan/rangkuman.",
                "Kinesthetic": "Kamu lebih mudah memahami materi lewat praktik langsung dan simulasi.",
            }
            st.info(label_desc.get(pred_label, ""))

            if hasattr(model_klasifikasi, "predict_proba"):
                try:
                    proba = model_klasifikasi.predict_proba(input_df)[0]
                    proba_df = pd.DataFrame({"Gaya Belajar": LE_CLASSES, "Probabilitas": proba})
                    proba_df = proba_df.sort_values("Probabilitas", ascending=False)
                    st.markdown("#### Probabilitas tiap kelas:")
                    st.bar_chart(proba_df.set_index("Gaya Belajar"))
                except Exception:
                    pass
        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")


# ============================================================
# MENU 2: PREDIKSI IPK
# ============================================================
else:
    st.title("\U0001F4CA Prediksi IPK Mahasiswa")
    st.markdown(
        "Isi data dan kebiasaan belajar kamu di bawah ini, lalu klik **Prediksi** "
        "untuk mengetahui estimasi IPK kumulatif kamu."
    )
    st.markdown("---")

    with st.form("form_regresi"):
        input_values = {}

        st.subheader("\U0001F464 Data Diri & Akademik")
        col1, col2 = st.columns(2)
        with col1:
            input_values[" Usia  (tahun)"] = st.number_input("Usia (tahun)", min_value=16, max_value=40, value=20)
        with col2:
            input_values[" IPS semester terakhir  "] = st.number_input(
                "IPS semester terakhir", min_value=0.0, max_value=4.0, value=3.0, step=0.01, format="%.2f"
            )

        st.markdown("##### Data Kategorik")
        for col in FITUR_KATEGORIKAL:
            options = KATEGORIKAL_VALUES.get(col, [])
            label = col.strip()
            if len(options) > 15:
                # Kolom dengan banyak nilai unik (mis. Program Studi free-text) -> selectbox + opsi ketik manual
                pilihan = st.selectbox(label, options + ["Lainnya (ketik manual)"], key=f"cat_{col}")
                if pilihan == "Lainnya (ketik manual)":
                    pilihan = st.text_input(f"Ketik manual untuk: {label}", value=options[0], key=f"cat_manual_{col}")
                input_values[col] = pilihan
            else:
                input_values[col] = st.selectbox(label, options, key=f"cat_{col}")

        st.markdown("---")
        st.subheader("\U0001F4DD Kuesioner Gaya Belajar, Kebiasaan Belajar & Lingkungan")
        st.caption("Skala 1 = Sangat Tidak Setuju, 5 = Sangat Setuju")

        likert_cols = [c for c in FITUR_NUMERIK if c not in LIKERT_EXCLUDE]
        for col in likert_cols:
            label = col.strip()
            input_values[col] = st.slider(label, 1, 5, 3, key=f"num_{col}")

        submitted2 = st.form_submit_button("\U0001F50D Prediksi IPK", use_container_width=True)

    if submitted2:
        try:
            row = {col: input_values[col] for col in X_COLUMNS_ORDER}
            input_df = pd.DataFrame([row], columns=X_COLUMNS_ORDER)

            pred_ipk = model_regresi.predict(input_df)[0]
            pred_ipk = float(np.clip(pred_ipk, 0.0, 4.0))

            st.markdown("---")
            st.success(f"### \U0001F3AF Estimasi IPK Kamu: **{pred_ipk:.2f}**")

            if pred_ipk >= 3.5:
                st.info("\U0001F31F Predikat: Cumlaude / Sangat Memuaskan")
            elif pred_ipk >= 3.0:
                st.info("\U0001F44D Predikat: Memuaskan")
            elif pred_ipk >= 2.5:
                st.info("\U0001F4C8 Predikat: Cukup Baik")
            else:
                st.warning("\u26A0\ufe0f Predikat: Perlu Peningkatan")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
            st.exception(e)
