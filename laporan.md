# Laporan Proyek Machine Learning
### Nama : Lia Fitriyanti
### Nim : 211351072
### Kelas : Pagi B

## Domain Proyek

Kanker prostat adalah bentuk kanker yang berkembang di kelenjar prostat.

Singkatnya:

Kanker pada prostat pria, kelenjar kecil seukuran kenari yang menghasilkan cairan mani. Prostat seorang pria menghasilkan cairan mani yang memelihara dan mengangkut sperma. Gejala termasuk kesulitan buang air kecil, tetapi kadang-kadang tidak ada gejala sama sekali.

  Referensi:
  Klasifikasi Penyakit Kanker Prostat Menggunakan Algoritma Naïve Bayes dan K-Nearest Neighbor

  https://ojs.unikom.ac.id/index.php/komputika/article/view/9629 

## Business Understanding

Kanker prostat berkembang ketika sel-sel abnormal di kelenjar prostat tumbuh tidak terkendali, membentuk tumor ganas. Diperkirakan lebih dari 24.200 pria didiagnosis menderita kanker prostat pada tahun 2022. Usia rata-rata saat diagnosis adalah 69 tahun.

### Problem Statements

- Ketidaktahuan masyarakat tentang gejala atau hal-hal yang bisa menyebabkan kanker prostat.

### Goals

- Memberikan informasi tentang gejala kanker prostat melalui aplikasi dengan cara menginputkan hal-hal yang menyebabkan kanker prostat dan mencegah ke stadium lanjut.


    ### Solution statements
    - Membuat platform klasifikasi kanker prostat berbasis web bagi masyarakat untuk memberikan informasi agar mencegah masyarakat ke stadium lanjut.
    - Model yang dihasilkan dari dataset itu menggunakan metode Logistic Regression.

## Data Understanding
Dataset yang saya gunakan berasal dari Kaggle tanda-tanda gejala kanker prostat. Dataset ini berisi 100 baris dan 10 kolom yang terdiri 8 kolom numerik, satu kolom kategori, dan satu kolom ID.

Prostate Cancer
https://www.kaggle.com/datasets/sajidsaifi/prostate-cancer 

### Variabel-variabel pada Heart Failure Prediction Dataset adalah sebagai berikut:
- id : ID pasien (int64)  
- diagnosis_result : hasil diagnosa dari inputan (object)
- radius : rata-rata jarak dari pusat ke titik perimeter (int64)  
- texture : tekstur (int64)  
- perimeter : perimeter (int64)  
- area : area (int64)  
- smoothness : kelancaran (float64)
- compactness : kekompakan (float64)
- symmetry : simetri (float64)
- fractal_dimension : dimensi fraktal (float64)

## Data Preparation
Pertama, import library yang akan digunakan.

    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score

Langkah selanjutnya, import file dataset yang akan digunakan. Berhubungan menggunakan vs code, kita tinggal import saja dengan catatan file dataset berada dalam satu folder dengan file jupyter.

    df = pd.read_csv("Prostate_Cancer.csv")

Kemudian melakukan transformasi data pada atribut diagnosis_result agar bisa diolah menggunakan Logisic Regression dan menyimpan hasil transformasi pada file dataset baru.

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['diagnosis_result'] = le.fit_transform(df['diagnosis_result'])

    df.to_csv('kanker_prostat.csv')

## Modeling
Model yang digunakan yaitu Logistic Regression. Pertama, import library yang akan digunakan.

    from sklearn.model_selection import  train_test_split
    from sklearn.linear_model import LogisticRegression

Kemudian seleksi fitur yang akan digunakan dan memilih atribut yang akan dijadikan sebagai label.

    X = df.drop(columns=['diagnosis_result', 'id'], axis=1)
    Y = df['diagnosis_result']

Selanjutnya, menentukan data yang akan dijadikan menjadi data training dan data testing. Data training berjumlah 80% dari total seluruh data sedangkan data testing berjumlah 20%.

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

## Evaluation
Dengan menggunakan algoritma Logistic Regression, didapatkan hasil akurasi dari data training sebesar 0.8375 sedangkan untuk akurasi data testing sebesar 0.9

## Deployment
https://klasifikasi-rb8bxxuhapv6kkaxnyrxmq.streamlit.app/

![Alt text](image.png)