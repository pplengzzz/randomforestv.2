import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(page_title='Water Level Prediction (RandomForest)', page_icon=':ocean:')

# ชื่อของแอป
st.title("การจัดการข้อมูลระดับน้ำและการพยากรณ์ด้วย RandomForest")

# อัปโหลดไฟล์ CSV
uploaded_file = st.file_uploader("เลือกไฟล์ CSV", type="csv")

# ฟังก์ชันสำหรับการอ่านข้อมูล
def read_and_clean_data(file_path):
    data = pd.read_csv(file_path)
    # ทำความสะอาดข้อมูลให้อยู่ในช่วงที่ต้องการ
    cleaned_data = data[(data['wl_up'] >= 100) & (data['wl_up'] <= 450)].copy()
    cleaned_data['datetime'] = pd.to_datetime(cleaned_data['datetime'])
    cleaned_data.set_index('datetime', inplace=True)
    return cleaned_data

# ฟังก์ชันสำหรับการเติมช่วงเวลาให้ครบทุก 15 นาที
def fill_missing_timestamps(data):
    full_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='15T')
    full_data = data.reindex(full_range)
    return full_data

# ฟังก์ชันสำหรับการเพิ่มฟีเจอร์ด้านเวลาและ lag features
def add_features(data):
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['minute'] = data.index.minute
    data['lag_1'] = data['wl_up'].shift(1)
    data['lag_2'] = data['wl_up'].shift(2)
    data['lag_1'].ffill(inplace=True)
    data['lag_2'].ffill(inplace=True)
    return data

# ฟังก์ชันสำหรับการเติมค่าด้วย RandomForestRegressor
def fill_missing_values(data):
    original_nan_indexes = data[data['wl_up'].isna()].index
    data['week'] = data.index.to_period("W").astype(str)
    missing_weeks = data[data['wl_up'].isna()]['week'].unique()
    filled_data = data.copy()

    for week in missing_weeks:
        week_data = data[data['week'] == week]
        missing_idx = week_data[week_data['wl_up'].isna()].index
        train_data = week_data.dropna(subset=['wl_up', 'hour', 'day_of_week', 'minute', 'lag_1', 'lag_2'])

        if len(train_data) > 1:
            X_train = train_data[['hour', 'day_of_week', 'minute', 'lag_1', 'lag_2']]
            y_train = train_data['wl_up']
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            X_missing = week_data.loc[missing_idx, ['hour', 'day_of_week', 'minute', 'lag_1', 'lag_2']]
            X_missing_clean = X_missing.dropna()

            if not X_missing_clean.empty:
                filled_values = model.predict(X_missing_clean)
                filled_data.loc[X_missing_clean.index, 'wl_up'] = filled_values

    filled_data['wl_up'].ffill(inplace=True)
    filled_data['wl_up'].bfill(inplace=True)
    return filled_data, original_nan_indexes

# ฟังก์ชันสำหรับการ plot ข้อมูล
def plot_filled_data(filled_data, original_nan_indexes):
    plt.figure(figsize=(14, 7))
    plt.plot(filled_data.index, filled_data['wl_up'], label='Actual Values', color='blue', alpha=0.6)
    filled_points = filled_data.loc[original_nan_indexes]
    plt.scatter(filled_points.index, filled_points['wl_up'], label='Filled Values', color='red', alpha=0.6)
    plt.title('Water Level Over Time with Filled Values')
    plt.xlabel('DateTime')
    plt.ylabel('Water Level (wl_up)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(plt)

# การประมวลผลหลังจากอัปโหลดไฟล์
if uploaded_file is not None:
    # อ่านไฟล์ CSV ที่อัปโหลด
    cleaned_data = read_and_clean_data(uploaded_file)

    # เติมช่วงเวลาให้ครบทุก 15 นาที
    full_data = fill_missing_timestamps(cleaned_data)

    # เพิ่มฟีเจอร์
    full_data = add_features(full_data)

    # เลือกช่วงวันที่จากผู้ใช้
    start_date = st.date_input("เลือกวันเริ่มต้น", pd.to_datetime(full_data.index.min()).date())
    end_date = st.date_input("เลือกวันสิ้นสุด", pd.to_datetime(full_data.index.max()).date())

    # แปลง start_date และ end_date เป็นชนิด datetime และลบ timezone เพื่อให้เป็น tz-naive
    start_date = pd.to_datetime(start_date).tz_localize(None)
    end_date = pd.to_datetime(end_date).tz_localize(None)

    # ตรวจสอบว่าช่วงวันที่เลือกถูกต้องหรือไม่
    if start_date < end_date:
        # กรองข้อมูลตามช่วงวันที่ที่เลือก และลบ timezone ของ cleaned_data เพื่อให้ตรงกัน
        selected_data = full_data.tz_localize(None).loc[start_date:end_date]

        # เติมค่าและเก็บตำแหน่งของ NaN เดิม
        filled_data, original_nan_indexes = fill_missing_values(selected_data)

        # พล๊อตผลลัพธ์การทำนายและข้อมูลจริง
        st.markdown("---")
        st.write("ทำนายระดับน้ำและเติมค่าในข้อมูลที่ขาดหาย")

        plot_filled_data(filled_data, original_nan_indexes)

        # แสดงผลลัพธ์การทำนายเป็นตาราง
        st.subheader('ตารางข้อมูลที่ทำนาย (datetime, code, wl_up)')
        st.write(filled_data[['code', 'wl_up']])

    else:
        st.error("กรุณาเลือกช่วงวันที่ที่ถูกต้อง (วันเริ่มต้นต้องน้อยกว่าวันสิ้นสุด)")
