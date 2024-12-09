import streamlit as st
import joblib
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder

brands = ['Lenovo', 'Dell', 'HP', 'Acer', 'ASUS', 'MSI', 'Samsung', 'Apple', 'Fujitsu', 'Microsoft', 'Gigabyte']
processors = ['AMD Hexa-Core Ryzen 5', 'Intel Core i5 (12th Gen)', 'Intel Core i5 (11th Gen)', 'Intel Core i3 (8th Gen)', 'AMD Octa-Core Ryzen 7', 'AMD Dual-Core Ryzen 3', 'Intel Core i3 (11th Gen)', 'Intel Core Ultra 5', 'Intel Celeron Dual-Core', 'AMD Quad-Core Ryzen 3', 'Intel Core i5 (10th Gen)', 'Intel Core i3 (12th Gen)', 'Intel Pentium Quad-Core', 'Intel Core i5 (13th Gen)', 'Intel Core i9 (11th Gen)', 'Intel Core i7 (10th Gen)', 'Intel Core i3 (10th Gen)', 'AMD Quad-Core Ryzen 5', 'Apple M1', 'Intel Core i5 (5th Gen)', 'Intel Core i3 (7th Gen)', 'Intel Core i5 (8th Gen)', 'Intel Pentium Dual-Core', 'Intel Core i5 (9th Gen)', 'Intel Core i3 (6th Gen)', 'AMD Dual-Core Athlon', 'Intel Core i7 (11th Gen)', 'Intel Core i7 (12th Gen)', 'Intel Core i7 (7th Gen)', 'Intel Core i7 (13th Gen)', 'Qualcomm Snapdragon Octa-Core', 'Intel Core i5 (7th Gen)', 'AMD Quad-Core APU', 'Intel Core i3 (13th Gen)', 'AMD Dual-Core A6 APU', 'AMD Octa-Core Ryzen 9', 'AMD Dual-Core A9 APU', 'Intel Core i7 (8th Gen)', 'Intel Core 7 (Series 1)', 'Intel Pentium Gold', 'Intel Core i9 (12th Gen)', 'Intel Core i5 (6th Gen)', 'Intel Core i7 (14th Gen)', 'Intel Core Ultra 7', 'Intel Core i9 (13th Gen)', 'Intel Core 5 (Series 1)', 'Intel Core Ultra 9', 'AMD Quad-Core Ryzen 7', 'Intel Core i7 (9th Gen)', 'Intel Atom Quad-Core', 'Intel Celeron Quad-Core', 'AMD Quad-Core A8 APU', 'AMD Dual-Core A4 APU', 'Intel Core i3 (5th Gen)', 'AMD Dual-Core E1 APU', 'AMD Quad-Core E2 APU', 'Intel Core i5 (1st Gen)', 'AMD Dual-Core APU', 'AMD Quad-Core A12 APU', 'AMD Quad-Core A6 APU', 'Intel Core M3 (6th Gen)', 'Intel Core i3 (4th Gen)', 'Intel Core i5 (4th Gen)', 'Intel Core i3 (3rd Gen)']
cpu_brands = ['AMD', 'Intel', 'Apple', 'Qualcomm']
ram = [8, 16, 4, 32, 12, 2, 24, 6]
ghzs = [4.0, 3.3, 4.2, 2.5, 2.2, 2.9, 3.4, 2.6, 1.7, 3.0, 3.1, 2.7, 2.8, 2.4, 2.1, 4.7, 4.8, 1.2, 1.0, 1.8, 2.0, 1.1, 2.3, 1.6, 0.8, 1.3, 1.5, 1.9, 5.5, 1.4]
display_types = ['LCD', 'LED']
display_sizes = ['15.6', '14', '17.3', '11.6', '13.3', '16.1', '16', '15.3', '14.1', '13.4', '15', '10.5', '13.5', '13', '12.4', '14.0', '17', '10.1', '15.', '14.9', '11', '12.3', '16.6', '14.5', '13.6', '12.0', '12', '16.2']
gpus = ['radeon', 'geforce 3050', 'iris xe', 'uhd 620', 'geforce 2050', 'geforce 4050', 'uhd', 'arc', 'geforce 1650', 'uhd 605', 'geforce 3060', 'geforce 3050 ti', 'integrated', 'radeon vega 3', 'uhd graphics', 'm1', 'hd 6000', 'uhd 600', 'radeon vega 6', 'hd 620', 'geforce mx150', 'geforce 1650 ti', 'radeon vega 8', 'hd 520', 'radeon rx 5500m', 'radeon 610m', 'radeon rx6500m', 'hd 500', 'radeon graphics', 'geforce 3070 ti', 'geforce 1050', 'radeon rx 6500m', 'radeon rx 5600m', 'geforce mx450', 'geforce 4060', 'adreno', 'geforce 1060', 'geforce mx330', 'radeon r2', 'radeon r4', 'iris xe graphics', 'geforce mx350', 'radeon rx 6800m', 'geforce mx250', 'radeon r5', 'radeon vega 7', 'mx350', 'geforce mx550', 'integrated iris xe', 'radeon 610m vega 2', 'geforce 3070', 'hd', 'iris plus', 'geforce mx570', 'geforce 940mx', 'radeon rx 6700s', 'intel iris xe', 'hd 615', 'geforce n16v-gmr1', 'arc a370m', 'uma', 'radeon rx6550m', 'radeon 780m', 'geforce 4070', 'radeon 760m', 'a500', 'integrated arc', 'radeon 680m', 'graphics', 'integrated hd', 'geforce p620', 'uhd 617', 'geforce gt 940mx', 'geforce 1050 ti', 'radeon rx vega 8', 'hd 405', 'radeon r16m-m1-30', 'geforce 1050t', 'radeon 520', 'hd 400', 'radeon 530', 'r17m-m1-70', 'geforce mx130', 'radeon hd 520', 'geforce 1660 ti', 'geforce 920mx', 'hd 610', 'radeon hd 8210', 'radeon rx 560x', 'integrated uhd', 'hd 505', 'radeon r3', 'geforce 2060', 'iris x', 'geforce mx110', 'radeon vega', 'hd 5500', 'radeon r5 m430', 'geforce', 'radeon rx vega 10', 'radeon vega 8 mobile', 'geforce mx 450', 'geforce mx230', 'radeon athlon 3000g', 'radeon rx 6600m', 'radeon rx6700m', 'arc a530m', 'quadro t550', 'quadro t600', 'radeon rx 7600s', 'geforce mx 350', 'uhd graphics 620', 't600', 'hd graphics', 'radeon r7', 'iris pro', 'integrated graphics', 'geforce mx 150', 'radeon r7 m260', 'geforce 920m', 'iris plus graphics 645', 'iris plus 655', 'iris plus 640', 'radeon rx vega 6', 'iris plus graphics 655', 'geforce 940m', 'geforce 930m', 'radeon hd 8670m', 'geforce 820m', 'radeon hd r7 m265', 'geforce 830m', 'hd 4000', 'geforce 930mx', 'geforce gt 740m', 'hd 4400', 'geforce gt 820m', 'radeon rx 580', 'iris xe max', 'geforce 940 mx', 'geforce a1000', 'geforce 1070', 'radeon r5 m330', 'hd graphics 620', 'radeon 535', 'geforce 1660', 'radeon rx640', 'radeon rx 640', 'radeon rx vega 5', 'radeon rx vega 7']
gpu_brands = ['AMD', 'Nvidia', 'Intel', 'Apple', 'Qualcomm']



neural_model = tf.keras.models.load_model('c:\\Kaggle laptop\\laptop_model.keras')
with open('random_forest_model.pkl', 'rb') as f:
    rfc_model = pickle.load(f)
with open('linear_regression_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)


st.title("Laptop Price Prediction")
st.header("Enter the specs")


in_brand = st.selectbox("Brand", options=brands)

in_processor_name = st.selectbox("Processor Name", options=processors)

in_ram_expandable = st.slider("RAM Expandable (GB)", 0, 64, 0)
in_ram = st.slider("RAM (GB)", 0, 64, 0)

in_ghz = st.number_input("Processor Speed (GHz)", min_value=0.0, max_value=5.0, step=0.1, format="%.1f")

in_display_type = st.selectbox("Display Type", options=display_types)
in_display_size = st.number_input("Display Size (inches)", min_value=10.0, max_value=20.0, step=0.1, format="%.1f")

in_gpu_brand = st.selectbox("GPU Brand", options=gpu_brands)
in_gpu = st.selectbox("GPU", options=gpus)

in_ssd = st.selectbox("SSD", [0, 16, 32, 64, 128, 256, 512, 1024])
in_hdd = st.selectbox("HDD", [0, 500, 1024, 2048])

in_adapter = st.slider("Adapter Power (W)", 0, 400, 0)


values = np.array([in_brand, in_processor_name, in_ram_expandable, in_ram, in_ghz, in_display_type, in_display_size, in_gpu, in_gpu_brand, in_ssd, in_hdd, in_adapter])
df = pd.DataFrame([values], columns=['Brand', 'Processor_Name', 'RAM_Expandable', 'RAM', 'Ghz', 'Display_type', 'Display', 'GPU', 'GPU_Brand', 'SSD', 'HDD', 'Adapter'])

cpu_encoder = joblib.load('cpu_encoder.pkl')
display_encoder = joblib.load('display_encoder.pkl')
gpu_encoder = joblib.load('gpu_encoder.pkl')
gpu_brand_encoder = joblib.load('gpu_brand_encoder.pkl')
brand_encoder = joblib.load('brand_encoder.pkl')

df['Processor_Name'] = cpu_encoder.transform(df['Processor_Name'])
df['Brand'] = brand_encoder.transform(df['Brand'])
df['GPU_Brand'] = gpu_brand_encoder.transform(df['GPU_Brand'])
df['GPU'] = gpu_encoder.transform(df['GPU'])
df['Display_type'] = display_encoder.transform(df['Display_type'])

scaler = joblib.load('scaler.pkl')
columns_to_scale = ['RAM', 'SSD', 'HDD', 'RAM_Expandable', 'Adapter']
df[columns_to_scale] = scaler.transform(df[columns_to_scale])


if st.button("Predict"):
    #neural_prediction = neural_model.predict(df)
    #linear_prediction = lr_model.predict(df)
    rfc_prediction = rfc_model.predict(df)
    #avg_prediction = (neural_prediction + linear_prediction + rfc_prediction) / 3
    #st.write(f"Predicted value: {4.6 * avg_prediction[0][0]:.2f} ft")
    st.write(f"Predicted price: {4.6 * rfc_prediction[0][0]:.2f} ft")