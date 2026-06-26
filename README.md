# INSTALASI.md

Panduan menjalankan DiabetaLens - Big Data Diabetes System* di komputer lokal. Project ini terdiri dari 2 bagian:

1. `Final.ipynb` - pipeline ETL (Extract-Transform-Load): membaca CSV mentah -> diproses dengan Apache Spark -> disimpan ke PostgreSQL (Supabase) dan MongoDB -> menghasilkan `dataset/dataset_diabetes_cleaned.csv`.
2. `app.py` - dashboard Streamlit yang membaca file CSV hasil ekspor dari langkah 1.

>  Jika Anda hanya ingin melihat dashboardnya (tanpa menjalankan ulang pipeline ETL), Anda bisa skip bagian database & Spark - cukup pastikan file `dataset/dataset_diabetes_cleaned.csv` sudah ada, lalu langsung ke Langkah 6.


## 1. Prasyarat


1. Python versi 3.9  3.11 
cara cek: `python --version`
2. Java (JDK) versi 11 atau 17
cara cek`java -version`
3. pip versi terbaru
cara cek `pip --version`

Catatan: PySpark butuh JVM. Jika belum ada Java, install dulu:
- Windows: download [Adoptium Temurin JDK 17](https://adoptium.net/) -> install -> set `JAVA_HOME` di Environment Variables.

Verifikasi `JAVA_HOME` sudah terbaca:
echo %JAVA_HOME%


## 2. Siapkan Folder Project

# letakkan app.py dan Final.ipynb di folder ini

Buat folder yang dibutuhkan:
1. dataset
2. images

Letakkan dataset mentah Anda dengan nama `dataset/raw_diabetes.csv`


## 3. Setup Virtual Environment & Dependencies

python -m venv venv

jalankan `requirements.txt`:

streamlit
pandas
numpy
plotly
pyspark==3.5.1
pymongo
psycopg2-binary
sqlalchemy
kaleido
jupyter
python-dotenv

command:
pip install -r requirements.txt


## 4. Setup Database #1 - PostgreSQL (Lokal, pengganti Supabase)

### a. Buat database baru
### b. Driver JDBC PostgreSQL untuk Spark
Tidak perlu diunduh manual. Pada `Final.ipynb`, Spark sudah diatur untuk mengunduh otomatis driver-nya dari Maven:
.config('spark.jars.packages', 'org.postgresql:postgresql:42.5.4')
Pastikan komputer terkoneksi internet saat pertama kali `SparkSession` dibuat.


## 5. Setup Database #2 - MongoDB (Lokal)

Notebook ini terhubung lewat:

MONGO_CONFIG = {'uri': 'mongodb://localhost:27017/', 'database': 'diabetes'}


### a. Install MongoDB Community Server
- Windows/macOS:  download dari [mongodb.com/try/download/community](https://www.mongodb.com/products/tools/compass?)

### b. Jalankan MongoDB sebagai service
# Windows: otomatis berjalan sebagai service setelah install

### c. Verifikasi aktif


## 6. Ringkasan Urutan Menjalankan dari Nol


# 1. python -m venv venv && source venv/bin/activate
# 2. pip install -r requirements.txt
# 3. mkdir dataset images
# 4. Jalankan ETL
# 5. Jalankan dashboard
     streamlit run app.py
