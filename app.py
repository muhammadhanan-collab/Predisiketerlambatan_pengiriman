import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediksi Keterlambatan pengiriman", layout="centered")

st.title("🚚 Prediksi Keterlambatan Pengiriman logistik")
st.write("Aplikasi prediksi risiko keterlambatan pengiriman ekspedisi")
st.write("Made by M Hamzah Hanan")


def load_model():
    return joblib.load("random_forest_model.pkl")

model = load_model()


# ============================
# FUNGSI ESTIMASI SHIPMENT COST (BERBASIS DATASET)
# ============================
def estimate_shipment_cost(distance_km, package_weight):
    cost_per_km = 127.55353170329201
    cost_per_kg = 13089.897081872845
    return (distance_km * cost_per_km) + (package_weight * cost_per_kg)



# ============================
# INPUT USER
# ============================
st.subheader("📦 Input Data Pengiriman")


col1, col2 = st.columns(2)


with col1:
 distance_km = st.number_input("Jarak Pengiriman (km)", min_value=1.0, value=300.0)
 package_weight = st.number_input("Berat Paket (kg)", min_value=0.1, value=10.0)
 warehouse_processing_time = st.number_input("Waktu Proses Gudang (jam)", min_value=0.0, value=4.0)


with col2:
 shipment_mode = st.selectbox("Moda Pengiriman", ["road", "sea", "air"])
 weather_condition = st.selectbox("Kondisi Cuaca", ["clear", "rain", "storm"])


# ============================
# PREDIKSI
# ============================
# ============================
# PREDIKSI
# ============================
if st.button("🔍 Prediksi"):

    # Hitung shipment cost otomatis
    shipment_cost = estimate_shipment_cost(distance_km, package_weight)

    st.info(f"💰 Estimasi Biaya Pengiriman: Rp {shipment_cost:,.0f}")

    # Data untuk model (HARUS SAMA DENGAN TRAINING)
    input_data = pd.DataFrame({
        "distance_km": [distance_km],
        "package_weight": [package_weight],
        "shipment_cost": [shipment_cost],
        "warehouse_processing_time": [warehouse_processing_time],
        "shipment_mode": [shipment_mode],
        "weather_condition": [weather_condition]
    })

    # Prediksi probabilitas keterlambatan
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Hasil Prediksi Risiko")

    if prob >= 0.6:
        st.error(f"🔴 Risiko Tinggi Terlambat ({prob:.2%})")
    elif prob >= 0.4:
        st.warning(f"🟠 Risiko Sedang Terlambat ({prob:.2%})")
    else:
        st.success(f"🟢 Risiko Rendah Terlambat ({prob:.2%})")

    st.subheader("📌 Rekomendasi Tindakan")

    if prob >= 0.6:
     st.markdown("""
    **🔴 Risiko Tinggi**
    - Pertimbangkan percepatan proses gudang  
    - Evaluasi perubahan moda pengiriman  
    - Tambahkan buffer waktu pengiriman  
    - Lakukan monitoring secara  intensif
    """)
    elif prob >= 0.4:
     st.markdown("""
    **🟠 Risiko Sedang**
    - Lakukan monitoring berkala  
    - Pastikan kesiapan armada dan dokumen  
    - Koordinasi dengan pihak gudang
    """)
    else:
     st.markdown("""
    **🟢 Risiko Rendah**
    - Pengiriman dapat dilakukan sesuai rencana  
    - Tetap lakukan monitoring standar
    """)

    st.write("---")
    st.caption("Prediksi bersifat probabilistik dan digunakan sebagai alat bantu pengambilan keputusan.")

