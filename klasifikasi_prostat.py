import pickle
import numpy as np
import streamlit as st

model = pickle.load(open('prostat.sav', 'rb'))

st.title('Klasifikasi Kanker Prostat')

radius = st.number_input('Radius')
texture = st.number_input('Tekstur')
perimeter = st.number_input('Perimeter')
area = st.number_input('Area')
smoothness = st.number_input('Kelancaran')
compactness = st.number_input('Kekompakan')
symmetry = st.number_input('Simetri')
fractal_dimension = st.number_input('Dimensi Fraktal')

prediksi = ''
if st.button('Hasil Prediksi'):
    prediksi = model.predict([[radius, texture, perimeter, area, smoothness, compactness,
                               symmetry, fractal_dimension]])

    if(prediksi[0] == 0):
        prediksi = 'Pasien mengidap kanker prostat jinak'
    else:
        prediksi = 'Pasien mengidap kanker prostat ganas'
st.success(prediksi)