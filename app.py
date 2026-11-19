import streamlit as st
import pandas as pd
import numpy as np
import gdown
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Dự đoán giá nhà King County - Nhóm 5",
    page_icon="🏠",
    layout="wide"
)

@st.cache_resource
def load_and_train():
    train_url = 'https://drive.google.com/uc?id=1EqzT0yb1pZBlJHx8x2-VKMcqkGH3KqWo'
    test_url = 'https://drive.google.com/uc?id=1tvYp2sOV6IOsEREZ4LJLVQsLb9fIhsKT'
    gdown.download(train_url, 'Nhom5_dulieugianha_train.csv', quiet=True)
    gdown.download(test_url, 'Nhom5_dulieugianha_test.csv', quiet=True)
    
    df_train = pd.read_csv('Nhom5_dulieugianha_train.csv')
    df_test = pd.read_csv('Nhom5_dulieugianha_test.csv')

    # Đủ 28 thuộc tính đầu vào X khớp chính xác Bảng 3.4.1 trong Báo cáo
    features = [
        'sqft_living', 'sqft_lot', 'bedrooms', 'bathrooms', 'floors', 'waterfront',
        'view', 'condition', 'grade', 'sqft_basement', 'yr_built', 'zipcode',
        'lat', 'long', 'sqft_living15', 'sqft_lot15', 'sale_year', 'sale_month',
        'sale_days', 'house_age', 'renovated', 'last_update_age', 'has_basement',
        'dist_center', 'total_rooms', 'living_lot_ratio', 'living_vs_nbr', 'lot_vs_nbr'
    ]
    imputer = SimpleImputer(strategy='median')
    X_train_clean = pd.DataFrame(imputer.fit_transform(df_train[features]), columns=features)
    model_lr = LinearRegression()
    model_lr.fit(X_train_clean, df_train['log_price'])

    # K-Means trên toàn bộ 29 cột số (trừ zipcode) khớp Mục 3.4.2 & 4.2
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    for col in ['id', 'date', 'zipcode']:
        if col in numeric_cols:
            numeric_cols.remove(col)

    X_kmeans_raw = df_train[numeric_cols].fillna(df_train[numeric_cols].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_kmeans_raw)
    kmeans_model = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
    df_train['Cluster'] = kmeans_model.fit_predict(X_scaled)

    cluster_means = df_train.groupby('Cluster')['price'].mean()
    cluster_names = {
        0: f"Cụm 0 – Nhà cao cấp, rộng rãi (Giá TB cụm: ${cluster_means[0]:,.0f} USD)",
        1: f"Cụm 1 – Nhà mới, tầm trung (Giá TB cụm: ${cluster_means[1]:,.0f} USD)",
        2: f"Cụm 2 – Nhà cũ, nhỏ, giá thấp (Giá TB cụm: ${cluster_means[2]:,.0f} USD)"
    }

    return df_test, features, imputer, model_lr, numeric_cols, scaler, kmeans_model, cluster_names

df_test, features, imputer, model_lr, numeric_cols, scaler, kmeans_model, cluster_names = load_and_train()

st.title("🏠 ỨNG DỤNG DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN KING COUNTY - NHÓM 5")
st.markdown(
    "**Thuật toán 1:** Linear Regression (28 biến đầu vào X, $R^2 = 0.8101$, MAE = `$97,650.62`) | "
    "**Thuật toán 2:** K-Means Clustering ($K = 3$ cụm) | "
    "**Nguồn dữ liệu:** Kaggle King County"
)
st.divider()

tab1, tab2 = st.tabs([
    "📊 1. Kiểm thử & Đối chiếu trên File Test", 
    "✍️ 2. Nhập đầy đủ 28 thuộc tính (X) để dự đoán"
])

# ================= TAB 1: KIỂM THỬ FILE TEST =================
with tab1:
    st.subheader("Đối chiếu trực tiếp kết quả trên tập kiểm thử (20% Test - 4.324 mẫu)")
    row_index = st.slider(
        "Chọn chỉ số dòng trong File Test (0 - 4323):", 
        min_value=0, max_value=len(df_test)-1, value=0, step=1
    )
    
    sample = df_test.iloc[[row_index]]
    X_sample = pd.DataFrame(imputer.transform(sample[features]), columns=features)
    
    pred_log = float(model_lr.predict(X_sample)[0])
    pred_price = float(np.exp(pred_log))
    real_price = float(sample['price'].values[0])
    diff = abs(real_price - pred_price)
    error_pct = (diff / real_price) * 100

    sample_km = sample.copy()
    sample_km['price'] = pred_price
    sample_km['log_price'] = pred_log
    cid = int(kmeans_model.predict(scaler.transform(sample_km[numeric_cols]))[0])

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Giá bán THỰC TẾ (File Test)", f"${real_price:,.2f} USD")
    col2.metric("🤖 Giá bán DỰ ĐOÁN (Linear Regression)", f"${pred_price:,.2f} USD")
    col3.metric("📉 Sai số tuyệt đối", f"${diff:,.2f} USD", f"{error_pct:.2f}%", delta_color="inverse")

    st.info(f"🏷️ **Kết quả Phân cụm K-Means (Mục 4.2):** {cluster_names[cid]}")

    st.markdown("**Toàn bộ 28 giá trị thuộc tính đầu vào (X) của mẫu đang chọn trong File Test (giữ nguyên giá trị đã biến đổi log):**")
    st.dataframe(sample[features], use_container_width=True)

# ================= TAB 2: NHẬP ĐẦY ĐỦ 28 BIẾN X (DẠNG SỐ THẬP PHÂN LOG KHỚP DỮ LIỆU) =================
with tab2:
    st.subheader("Nhập đầy đủ 28 thuộc tính đầu vào (X) khớp Bảng 3.4.1 & Dữ liệu đã tiền xử lý")
    st.caption("Lưu ý: Các thuộc tính diện tích (sqft_living, sqft_lot, sqft_basement, sqft_living15, sqft_lot15) sử dụng giá trị thập phân đã biến đổi log1p giống hệt trong tập dữ liệu huấn luyện.")
    
    c1, c2, c3, c4 = st.columns(4)
    
    # Cột 1: 7 thuộc tính (1 -> 7)
    with c1:
        st.markdown("**Nhóm 1: Diện tích & Cấu trúc (1–7)**")
        sqft_living = st.number_input("1. sqft_living (log)", min_value=0.0, value=7.555382, format="%.6f")
        sqft_lot = st.number_input("2. sqft_lot (log)", min_value=0.0, value=8.936035, format="%.6f")
        bedrooms = st.number_input("3. bedrooms (Số phòng ngủ)", min_value=1, max_value=15, value=3, step=1)
        bathrooms = st.number_input("4. bathrooms (Số phòng tắm)", min_value=0.5, max_value=10.0, value=2.25, step=0.25)
        floors = st.number_input("5. floors (Số tầng)", min_value=1.0, max_value=4.0, value=1.5, step=0.5)
        waterfront = st.selectbox("6. waterfront (Giáp mặt nước)", options=[0, 1], index=0)
        view = st.slider("7. view (Tầm nhìn 0–4)", 0, 4, 0)

    # Cột 2: 7 thuộc tính (8 -> 14)
    with c2:
        st.markdown("**Nhóm 2: Chất lượng & Vị trí (8–14)**")
        condition = st.slider("8. condition (Tình trạng 1–5)", 1, 5, 3)
        grade = st.slider("9. grade (Chất lượng xây 1–13)", 1, 13, 7)
        sqft_basement = st.number_input("10. sqft_basement (log)", min_value=0.0, value=0.0, format="%.6f")
        yr_built = st.number_input("11. yr_built (Năm xây dựng)", min_value=1900, max_value=2026, value=1975, step=1)
        zipcode = st.number_input("12. zipcode (Mã bưu chính)", value=98065, step=1)
        lat = st.number_input("13. lat (Vĩ độ)", value=47.5714, format="%.4f")
        long = st.number_input("14. long (Kinh độ)", value=-122.2310, format="%.4f")

    # Cột 3: 7 thuộc tính (15 -> 21)
    with c3:
        st.markdown("**Nhóm 3: Lân cận & Thời gian (15–21)**")
        sqft_living15 = st.number_input("15. sqft_living15 (log)", min_value=0.0, value=7.518064, format="%.6f")
        sqft_lot15 = st.number_input("16. sqft_lot15 (log)", min_value=0.0, value=8.938663, format="%.6f")
        sale_year = st.number_input("17. sale_year (Năm bán)", min_value=2014, max_value=2026, value=2014, step=1)
        sale_month = st.slider("18. sale_month (Tháng bán)", 1, 12, 6)
        sale_days = st.number_input("19. sale_days (Số ngày từ 01/05/2014)", min_value=0, value=168, step=1)
        house_age = st.number_input("20. house_age (Tuổi căn nhà)", min_value=0, value=40, step=1)
        renovated = st.selectbox("21. renovated (Đã cải tạo 0/1)", options=[0, 1], index=0)

    # Cột 4: 7 thuộc tính (22 -> 28)
    with c4:
        st.markdown("**Nhóm 4: Đặc trưng dẫn xuất (22–28)**")
        last_update_age = st.number_input("22. last_update_age (Tuổi từ lần sửa cuối)", min_value=0, value=37, step=1)
        has_basement = st.selectbox("23. has_basement (Có tầng hầm 0/1)", options=[0, 1], index=0)
        dist_center = st.number_input("24. dist_center (Khoảng cách TT - km)", min_value=0.0, value=16.514052, format="%.6f")
        total_rooms = st.number_input("25. total_rooms (Tổng số phòng)", min_value=1.0, value=5.0, step=0.25)
        living_lot_ratio = st.number_input("26. living_lot_ratio (Tỷ lệ nhà/đất)", min_value=0.0, value=0.248390, format="%.6f")
        living_vs_nbr = st.number_input("27. living_vs_nbr (So sánh DT nhà)", min_value=0.0, value=1.0, format="%.6f")
        lot_vs_nbr = st.number_input("28. lot_vs_nbr (So sánh DT đất)", min_value=0.0, value=1.0, format="%.6f")

    if st.button("🚀 DỰ ĐOÁN GIÁ NHÀ & PHÂN CỤM (ĐỦ 28 BIẾN X)", type="primary", use_container_width=True):
        # Đưa trực tiếp các giá trị đã biến đổi log vào mô hình (không log1p lại)
        row = {
            'sqft_living': sqft_living,
            'sqft_lot': sqft_lot,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'floors': floors,
            'waterfront': waterfront,
            'view': view,
            'condition': condition,
            'grade': grade,
            'sqft_basement': sqft_basement,
            'yr_built': yr_built,
            'zipcode': zipcode,
            'lat': lat,
            'long': long,
            'sqft_living15': sqft_living15,
            'sqft_lot15': sqft_lot15,
            'sale_year': sale_year,
            'sale_month': sale_month,
            'sale_days': sale_days,
            'house_age': house_age,
            'renovated': renovated,
            'last_update_age': last_update_age,
            'has_basement': has_basement,
            'dist_center': dist_center,
            'total_rooms': total_rooms,
            'living_lot_ratio': living_lot_ratio,
            'living_vs_nbr': living_vs_nbr,
            'lot_vs_nbr': lot_vs_nbr
        }

        input_df = pd.DataFrame([row])[features]
        input_imputed = pd.DataFrame(imputer.transform(input_df), columns=features)
        pred_log = float(model_lr.predict(input_imputed)[0])
        pred_price = float(np.exp(pred_log))

        km_row = row.copy()
        km_row['price'] = pred_price
        km_row['log_price'] = pred_log
        km_df = pd.DataFrame([km_row])[numeric_cols]
        cid = int(kmeans_model.predict(scaler.transform(km_df))[0])

        r1, r2 = st.columns(2)
        r1.success(f"### 💰 GIÁ DỰ ĐOÁN: **${pred_price:,.2f} USD**")
        r2.info(f"### 🏷️ PHÂN CỤM K-MEANS:\n**{cluster_names[cid]}**")
