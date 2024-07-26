import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Memuat model terbaik
model = joblib.load('best_model.pkl')

# Memuat data untuk pengkodean dan penskalaan
data = pd.read_csv('onlinefoods.csv')

# Daftar kolom yang diperlukan selama pelatihan
required_columns = ['Age', 'Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications', 'Family size', 'latitude', 'longitude', 'Pin code']

# Pastikan hanya kolom yang diperlukan ada
data = data[required_columns]

# Pra-pemrosesan data
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = data[column].astype(str)
    le.fit(data[column])
    data[column] = le.transform(data[column])
    label_encoders[column] = le

scaler = StandardScaler()
numeric_features = ['Age', 'Family size', 'latitude', 'longitude', 'Pin code']
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Fungsi untuk memproses input pengguna
def preprocess_input(user_input):
    processed_input = {col: [user_input.get(col, 'Unknown')] for col in required_columns}
    for column in label_encoders:
        if column in processed_input:
            input_value = processed_input[column][0]
            if input_value in label_encoders[column].classes_:
                processed_input[column] = label_encoders[column].transform([input_value])
            else:
                # Jika nilai tidak dikenal, berikan nilai default seperti -1
                processed_input[column] = [-1]
    processed_input = pd.DataFrame(processed_input)
    processed_input[numeric_features] = scaler.transform(processed_input[numeric_features])
    return processed_input

# CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #f5f5f5;
    }
    .stApp {
        background-color: #ffffff;
        padding: 2em;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    h3 {
        color: #34495e;
        font-size: 1.5em;
        margin-bottom: 1em;
    }
    .stButton>button {
        background-color: #2980b9;
        color: #ffffff;
        border: none;
        padding: 0.75em 1.5em;
        border-radius: 5px;
        cursor: pointer;
        font-size: 1em;
    }
    .stButton>button:hover {
        background-color: #1f6391;
    }
    .stNumberInput, .stSelectbox {
        margin-bottom: 1em;
    }
    .prediction-result {
        color: #ffffff;
        background-color: #2980b9;
        padding: 1em;
        border-radius: 5px;
        text-align: center;
        font-size: 1.25em;
        margin-top: 1em;
    }
    </style>
""", unsafe_allow_html=True)

# Antarmuka Streamlit
st.title("Prediksi Kepuasan Pelanggan Makanan Online")

st.markdown("<h3>Masukkan Data Pelanggan</h3>", unsafe_allow_html=True)

# Input pengguna
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Umur', min_value=18, max_value=100)
    gender = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
    marital_status = st.selectbox('Status Perkawinan', ['Lajang', 'Menikah'])
    occupation = st.selectbox('Pekerjaan', ['Pelajar', 'Pegawai', 'Wiraswasta'])

with col2:
    monthly_income = st.selectbox('Pendapatan Bulanan', ['Tidak Berpenghasilan', 'Di Bawah Rp10,000,000', 'Rp10,000,000 - Rp25,000,000', 'Rp25,000,000 - Rp50,000,000', 'Lebih dari Rp50,000,000'])
    educational_qualifications = st.selectbox('Pendidikan Terakhir', ['Sarjana Muda', 'Sarjana', 'Pasca Sarjana'])
    family_size = st.number_input('Ukuran Keluarga', min_value=1, max_value=20)
    latitude = st.number_input('Lintang', format="%f")
    longitude = st.number_input('Bujur', format="%f")
    pin_code = st.number_input('Kode Pos', min_value=100000, max_value=999999)

user_input = {
    'Age': age,
    'Gender': gender,
    'Marital Status': marital_status,
    'Occupation': occupation,
    'Monthly Income': monthly_income,
    'Educational Qualifications': educational_qualifications,
    'Family size': family_size,
    'latitude': latitude,
    'longitude': longitude,
    'Pin code': pin_code
}

if st.button('Prediksi'):
    user_input_processed = preprocess_input(user_input)
    try:
        prediction = model.predict(user_input_processed)
        st.markdown(f'<div class="prediction-result">Prediksi: {prediction[0]}</div>', unsafe_allow_html=True)
    except ValueError as e:
        st.error(f"Terjadi kesalahan dalam prediksi: {e}")

# Tambahkan elemen HTML untuk output
st.markdown("""
    <h3>Hasil Prediksi</h3>
    <p>Hasil prediksi akan ditampilkan di sini.</p>
""", unsafe_allow_html=True)
