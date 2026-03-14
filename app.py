import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import date, timedelta

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Stock Prediction System", page_icon="📈", layout="wide")
st.title("📈 ระบบวิเคราะห์และพยากรณ์แนวโน้มราคาหุ้น")
st.markdown("โปรเจกต์วิเคราะห์ข้อมูลและพยากรณ์ราคาหุ้นด้วย Machine Learning")

# 2. แถบด้านข้าง (Sidebar) สำหรับรับค่าจากผู้ใช้
st.sidebar.header("ตั้งค่าพารามิเตอร์")
ticker_symbol = st.sidebar.text_input("สัญลักษณ์หุ้น (เช่น AAPL, PTT.BK, TSLA)", "AAPL")
start_date = st.sidebar.date_input("วันที่เริ่มต้น", date.today() - timedelta(days=365))
end_date = st.sidebar.date_input("วันที่สิ้นสุด", date.today())
predict_days = st.sidebar.slider("จำนวนวันที่ต้องการพยากรณ์ล่วงหน้า", 1, 30, 7)

# 3. ฟังก์ชันดึงข้อมูลหุ้น
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('กำลังโหลดข้อมูล...')
data = load_data(ticker_symbol, start_date, end_date)
data_load_state.text('โหลดข้อมูลสำเร็จ! 🎉')

# 4. แสดงข้อมูลดิบ (Dataframe)
st.subheader(f"ข้อมูลหุ้น {ticker_symbol} ย้อนหลัง")
st.dataframe(data.tail())

# 5. สร้างกราฟแท่งเทียน (Candlestick Chart) ด้วย Plotly
st.subheader("📊 กราฟวิเคราะห์ราคาหุ้น")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=data['Date'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Market Data'))
fig.update_layout(xaxis_rangeslider_visible=False, height=500, margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig, use_container_width=True)

# 6. ส่วนของ Machine Learning (พยากรณ์แนวโน้ม)
st.subheader("🤖 พยากรณ์แนวโน้มด้วย Machine Learning (Linear Regression)")

# เตรียมข้อมูลสำหรับ ML
# เปลี่ยนวันที่เป็นตัวเลข (Ordinal) เพื่อใช้เทรนโมเดล
data['Date_Ordinal'] = pd.to_datetime(data['Date']).map(pd.Timestamp.toordinal)
X = data[['Date_Ordinal']]
y = data['Close']

# สร้างและเทรนโมเดล
model = LinearRegression()
model.fit(X, y)

# สร้างข้อมูลวันที่สำหรับอนาคต
future_dates = [data['Date'].iloc[-1] + timedelta(days=i) for i in range(1, predict_days + 1)]
future_dates_ordinal = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)

# ทำนายราคา
future_predict = model.predict(future_dates_ordinal)

# แสดงผลการพยากรณ์
pred_df = pd.DataFrame({'วันที่พยากรณ์': [d.strftime('%Y-%m-%d') for d in future_dates], 'ราคาปิดที่คาดการณ์ (USD/THB)': future_predict})
st.table(pred_df)

st.info("หมายเหตุ: นี่คือ Prototype สำหรับแสดงผลการทำงาน (Proof of Concept) โมเดลที่ใช้เป็นเพียง Linear Regression พื้นฐานสำหรับการสาธิตเท่านั้น")