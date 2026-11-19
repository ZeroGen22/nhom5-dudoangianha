import streamlit as st
import pandas as pd
import numpy as np
import gdown
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Định giá & Phân khúc BĐS King County - Nhóm 5",
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

    # 1. Huấn luyện mô hình Dự đoán giá (Linear Regression)
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

    # 2. Huấn luyện mô hình Phân cụm phân khúc (K-Means k=3)
    cluster_cols = ['price', 'sqft_living', 'grade', 'house_age', 'bedrooms', 'bathrooms']
    scaler = StandardScaler()
    X_cluster_scaled = scaler.fit_transform(df_train[cluster_cols])
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
    df_train['Cluster'] = kmeans.fit_predict(X_cluster_scaled)

    # Tự động gán nhãn phân khúc theo mức giá trung bình của từng cụm
    cluster_means = df_train.groupby('Cluster')['price'].mean().sort_values()
    sorted_ids = cluster_means.index.tolist()
    cluster_labels = {
        sorted_ids[0]: ("Phân khúc Bình dân / Nhà cũ diện tích nhỏ", cluster_means.iloc[0]),
        sorted_ids[1]: ("Phân khúc Tầm trung / Nhà mới hiện đại", cluster_means.iloc[1]),
        sorted_ids[2]: ("Phân khúc Cao cấp / Biệt thự rộng rãi", cluster_means.iloc[2])
    }
    
    return df_test, features, imputer, model_lr, cluster_cols, scaler, kmeans, cluster_labels

df_test, features, imputer, model_lr, cluster_cols, scaler, kmeans, cluster_labels = load_and_train()

st.title("🏠 HỆ THỐNG ĐỊNH GIÁ & PHÂN KHÚC BẤT ĐỘNG SẢN KING COUNTY - NHÓM 5")
st.markdown(
    "**Thuật toán 1:** Linear Regression ($R^2 = 0.8101$) | "
    "**Thuật toán 2:** K-Means Clustering ($K = 3$ phân khúc) | "
    "**Nguồn dữ liệu:** Kaggle King County (`Nhom5_DuDoanGiaNha_test`)"
)
st.divider()

tab1, tab2 = st.tabs([
    "📊 1. Kiểm thử & Đối chiếu trên File Test", 
    "✍️ 2. Định giá & Phân loại phân khúc nhà mới"
])

# ================= TAB 1: KIỂM THỬ FILE TEST =================
with tab1:
    st.subheader("Đối chiếu trực tiếp giữa Giá thực tế và Giá mô hình dự đoán trên tập Test")
    row_index = st.slider(
        "Chọn chỉ số dòng trong File Test (0 - 4323):", 
        min_value=0, max_value=len(df_test)-1, value=0, step=1
    )
    
    sample = df_test.iloc[[row_index]]
    X_sample = pd.DataFrame(imputer.transform(sample[features]), columns=features)
    
    pred_price = float(np.exp(model_lr.predict(X_sample)[0]))
    real_price = float(sample['price'].values[0])
    diff = abs(real_price - pred_price)
    error_pct = (diff / real_price) * 100

    # Xác định phân khúc K-means cho mẫu Test
    sample_cluster_input = pd.DataFrame([[
        pred_price, sample['sqft_living'].values[0], sample['grade'].values[0],
        sample['house_age'].values[0], sample['bedrooms'].values[0], sample['bathrooms'].values[0]
    ]], columns=cluster_cols)
    cid = int(kmeans.predict(scaler.transform(sample_cluster_input))[0])
    seg_name, seg_avg = cluster_labels[cid]

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Giá bán THỰC TẾ (File Test)", f"${real_price:,.2f} USD")
    col2.metric("🤖 Giá bán DỰ ĐOÁN (Linear Regression)", f"${pred_price:,.2f} USD")
    col3.metric("📉 Chênh lệch & Sai số", f"${diff:,.2f} USD", f"{error_pct:.2f}%", delta_color="inverse")

    st.info(f"🏷️ **Xếp loại phân khúc (K-Means):** Căn nhà thuộc **{seg_name}** (Mức giá trung bình phân khúc này: **${seg_avg:,.0f} USD**)")

    sqft_liv_real = np.expm1(sample['sqft_living'].values[0])
    sqft_lot_real = np.expm1(sample['sqft_lot'].values[0])
    
    st.markdown("**Thông tin chi tiết căn nhà trích xuất từ tập Test:**")
    info_df = pd.DataFrame({
        "Thuộc tính": ["Diện tích sử dụng", "Diện tích đất", "Phòng ngủ", "Phòng tắm", "Số tầng", "Chất lượng (Grade)", "Năm xây dựng", "Mã vùng (Zipcode)"],
        "Giá trị thực tế": [
            f"{sqft_liv_real:,.0f} sqft", f"{sqft_lot_real:,.0f} sqft",
            int(sample['bedrooms'].values[0]), float(sample['bathrooms'].values[0]),
            float(sample['floors'].values[0]), f"{int(sample['grade'].values[0])}/13",
            int(sample['yr_built'].values[0]), int(sample['zipcode'].values[0])
        ]
    })
    st.table(info_df)

# ================= TAB 2: NHẬP DỮ LIỆU MỚI =================
with tab2:
    st.subheader("Nhập thông số căn nhà để định giá & nhận diện phân khúc thị trường")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sqft_living = st.number_input("Diện tích sử dụng (sqft)", min_value=100, value=1910, step=50)
        sqft_lot = st.number_input("Diện tích đất (sqft)", min_value=200, value=7620, step=100)
        bedrooms = st.slider("Số phòng ngủ", 1, 10, 3)
        bathrooms = st.slider("Số phòng tắm", 0.5, 6.0, 2.25, step=0.25)
        floors = st.slider("Số tầng", 1.0, 3.5, 1.5, step=0.5)
    with c2:
        grade = st.slider("Chất lượng xây dựng Grade (1-13)", 1, 13, 7)
        condition = st.slider("Tình trạng nhà Condition (1-5)", 1, 5, 3)
        yr_built = st.number_input("Năm xây dựng", min_value=1900, max_value=2026, value=1975)
        renovated = st.selectbox("Đã cải tạo chưa?", options=[0, 1], format_func=lambda x: "0 - Chưa cải tạo" if x == 0 else "1 - Đã cải tạo")
        view = st.slider("Điểm tầm nhìn View (0-4)", 0, 4, 0)
    with c3:
        zipcode = st.number_input("Mã bưu chính (Zipcode)", value=98065)
        lat = st.number_input("Vĩ độ (Lat)", value=47.5714, format="%.4f")
        long = st.number_input("Kinh độ (Long)", value=-122.2310, format="%.4f")
        waterfront = st.selectbox("Giáp mặt nước?", options=[0, 1], format_func=lambda x: "0 - Không" if x == 0 else "1 - Có giáp mặt nước")
        sqft_basement = st.number_input("Diện tích tầng hầm (sqft)", min_value=0, value=0, step=50)
    with c4:
        sqft_living15 = st.number_input("DT sử dụng TB 15 nhà lân cận", min_value=100, value=1840)
        sqft_lot15 = st.number_input("DT đất TB 15 nhà lân cận", min_value=200, value=7620)
        sale_year = st.number_input("Năm giao dịch", min_value=2014, max_value=2026, value=2014)
        sale_month = st.slider("Tháng giao dịch", 1, 12, 6)

    if st.button("🚀 ĐỊNH GIÁ & PHÂN LOẠI PHÂN KHÚC", type="primary", use_container_width=True):
        sale_days = (pd.Timestamp(year=int(sale_year), month=int(sale_month), day=15) - pd.Timestamp("2014-05-01")).days
        house_age = max(0, int(sale_year) - int(yr_built))
        last_update_age = house_age if renovated == 0 else max(0, house_age // 2)
        has_basement = 1 if sqft_basement > 0 else 0
        dist_center = np.sqrt(((lat - 47.6062) * 111) ** 2 + ((long - (-122.3321)) * 75) ** 2)
        total_rooms = bedrooms + bathrooms
        living_lot_ratio = sqft_living / max(sqft_lot, 1)
        living_vs_nbr = sqft_living / max(sqft_living15, 1)
        lot_vs_nbr = sqft_lot / max(sqft_lot15, 1)

        row = {
            'sqft_living': np.log1p(sqft_living), 'sqft_lot': np.log1p(sqft_lot),
            'bedrooms': bedrooms, 'bathrooms': bathrooms, 'floors': floors,
            'waterfront': waterfront, 'view': view, 'condition': condition, 'grade': grade,
            'sqft_basement': np.log1p(sqft_basement), 'yr_built': yr_built, 'zipcode': zipcode,
            'lat': lat, 'long': long, 'sqft_living15': np.log1p(sqft_living15),
            'sqft_lot15': np.log1p(sqft_lot15), 'sale_year': sale_year, 'sale_month': sale_month,
            'sale_days': sale_days, 'house_age': house_age, 'renovated': renovated,
            'last_update_age': last_update_age, 'has_basement': has_basement,
            'dist_center': dist_center, 'total_rooms': total_rooms,
            'living_lot_ratio': living_lot_ratio, 'living_vs_nbr': living_vs_nbr, 'lot_vs_nbr': lot_vs_nbr
        }

        input_df = pd.DataFrame([row])[features]
        input_imputed = pd.DataFrame(imputer.transform(input_df), columns=features)
        pred_price = float(np.exp(model_lr.predict(input_imputed)[0]))

        # Phân cụm K-means cho căn nhà mới
        new_cluster_input = pd.DataFrame([[
            pred_price, np.log1p(sqft_living), grade, house_age, bedrooms, bathrooms
        ]], columns=cluster_cols)
        cid = int(kmeans.predict(scaler.transform(new_cluster_input))[0])
        seg_name, seg_avg = cluster_labels[cid]

        r1, r2 = st.columns(2)
        r1.success(f"### 💰 GIÁ DỰ ĐOÁN: **${pred_price:,.2f} USD**")
        r2.info(f"### 🏷️ PHÂN KHÚC: **{seg_name}**\n*(Giá TB phân khúc: ${seg_avg:,.0f} USD)*")
