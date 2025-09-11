import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Customer Segmentation", layout="wide")

# ===============================
# Load Data
# ===============================
@st.cache_data
def load_data():
    df_products = pd.read_csv("Products_with_Categories.csv")
    df_trans = pd.read_csv("Transactions.csv")
    return df_products, df_trans

df_products, df_trans = load_data()

# ===============================
# Compute RFM scores automatically
# ===============================
def compute_rfm_scores(df, user_id_col="Member_number", date_col="Date", prod_col="productId", amount_col=None):
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    snapshot_date = df[date_col].max() + pd.Timedelta(days=1)

    # Frequency = number of purchases
    freq = df.groupby(user_id_col)[prod_col].count()

    # Monetary = sum of amount if available, else fallback = count
    if amount_col and amount_col in df.columns:
        monetary = df.groupby(user_id_col)[amount_col].sum()
    else:
        monetary = freq.copy()

    # Recency = days since last purchase
    rec = df.groupby(user_id_col)[date_col].max().apply(lambda x: (snapshot_date - x).days)

    rfm = pd.DataFrame({"Recency": rec, "Frequency": freq, "Monetary": monetary})

    # Score each into [1–4] quartiles
    rfm["R"] = pd.qcut(rfm["Recency"], 4, labels=[4,3,2,1]).astype(int)
    rfm["F"] = pd.qcut(rfm["Frequency"], 4, labels=[1,2,3,4]).astype(int)
    rfm["M"] = pd.qcut(rfm["Monetary"], 4, labels=[1,2,3,4]).astype(int)

    return rfm.reset_index()

try:
    rfm_scores = compute_rfm_scores(df_trans)
except Exception as e:
    st.warning(f"Could not compute RFM automatically: {e}")
    rfm_scores = None

# ===============================
# Classification into 6 segments
# ===============================
def classify_customer(R, F, M):
    if R == 4 and F == 4 and M == 4:
        return "Champions"
    elif R >= 3 and F >= 3:
        return "Loyal Customers"
    elif F == 4 and R >= 2:
        return "Frequent Buyers"
    elif M == 4 and R >= 2:
        return "Big Spenders"
    elif R == 2 and F <= 2 and M <= 2:
        return "At Risk"
    elif R == 1:
        return "Lost Customers"
    else:
        return "Regular"

# ===============================
# Sidebar Menu
# ===============================
menu = st.sidebar.radio(
    "Menu",
    [   "Introduction",
        "Business Problem",
        "Evaluation & Report",
        "New Prediction / Analysis",
        "Recommendation"
        
    ]
)

# ===============================
# RFM Overview Image (Scatter)
# ===============================
# if rfm_scores is not None:
#     fig, ax = plt.subplots(figsize=(6,4))
#     sns.scatterplot(
#         data=rfm_scores,
#         x="Frequency", y="Monetary", size="Recency",
#         sizes=(20, 200), alpha=0.5, ax=ax
#     )
#     ax.set_title("RFM Overview")
#     st.pyplot(fig)
#     plt.close(fig)


# ===============================
# Business Problem
# ===============================
if menu == "Business Problem":
    st.title("📌 Business Problem")
    st.write("""
    Cửa hàng X chủ yếu bán các sản phẩm thiết yếu cho khách hàng
     như rau, củ, quả, thịt, cá, trứng, sữa, nước giải khát... 
    Khách hàng của cửa hàng là khách hàng mua lẻ.
    """)
    st.write("""->Chủ cửa hàng X mong muốn có thể bán được nhiều hàng hóa hơn
     cũng như giới thiệu sản phẩm đến đúng đối tượng khách hàng, 
    chăm sóc và làm hài lòng khách hàng.
    """)

# ===============================
# Evaluation & Report
# ===============================
elif menu == "Evaluation & Report":
    st.title("📊 Evaluation & Report")
    
    if rfm_scores is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        sns.countplot(x="R", data=rfm_scores, ax=axes[0])
        axes[0].set_title("Recency Score")
        sns.countplot(x="F", data=rfm_scores, ax=axes[1])
        axes[1].set_title("Frequency Score")
        sns.countplot(x="M", data=rfm_scores, ax=axes[2])
        axes[2].set_title("Monetary Score")
        st.pyplot(fig)

        rfm_scores["Segment"] = rfm_scores.apply(lambda x: classify_customer(x["R"], x["F"], x["M"]), axis=1)
        seg_counts = rfm_scores["Segment"].value_counts()

        st.bar_chart(seg_counts)
        st.dataframe(rfm_scores.head())
    else:
        st.warning("Không có dữ liệu RFM để hiển thị.")

# ===============================
# New Prediction (manual input)
# ===============================
elif menu == "New Prediction / Analysis":
    st.title("🔮 Customer Prediction (Manual Input)")

    col1, col2, col3 = st.columns(3)
    with col1:
        R = st.slider("Recency Score (1–4)", 1, 4, 2)
    with col2:
        F = st.slider("Frequency Score (1–4)", 1, 4, 2)
    with col3:
        M = st.slider("Monetary Score (1–4)", 1, 4, 2)

    if st.button("Predict"):
        segment = classify_customer(R, F, M)
        st.success(f"🏷️ This customer belongs to: **{segment}**")
    st.markdown("---")
    st.title("📂 Bulk Prediction (Upload file)")
    st.write("Tải lên một tệp CSV/Excel với các cột `R`, `F`, `M` để phân loại khách hàng.")

    file = st.file_uploader("Tải lên CSV hoặc Excel", type=["csv","xlsx"])
    if file:
        if file.name.endswith(".csv"):
            df_input = pd.read_csv(file)
        else:
            df_input = pd.read_excel(file)

        if {"R","F","M"}.issubset(df_input.columns):
            df_input["Segment"] = df_input.apply(lambda x: classify_customer(x["R"], x["F"], x["M"]), axis=1)
            st.dataframe(df_input.head(20))
            st.download_button(
                "Tải xuống dự đoán",
                data=df_input.to_csv(index=False).encode("utf-8"),
                file_name="rfm_predictions.csv",
                mime="text/csv"
            )
        else:
            st.error("File phải chứa các cột `R`, `F`, `M`.")

# ===============================
# Recommendation Section
# ===============================
elif menu == "Recommendation":
    st.title("💡 Recommendations by Segment")
    st.write("""
    - **Champions** → Reward with loyalty programs, early access.  
    - **Loyal Customers** → Exclusive deals to maintain engagement.  
    - **Frequent Buyers** → Suggest bundles, cross-selling.  
    - **Big Spenders** → Premium offers, VIP services.  
    - **At Risk** → Send reminders, discounts to reactivate.  
    - **Lost Customers** → Win-back campaigns with special offers.  
    """)

# ===============================
# Thông tin nhóm
# ===============================
elif menu == "Introduction":
    st.title("👨‍💻 Thông tin nhóm H")
    st.write("""
    - **Tên**: Trần Nhật Minh   
    - **Email**: nhatminhtr233@gmail.com   
    - **Project**: Customer Segmentation
    """)
    st.image("RFM_clustering.png", caption="RFM Clustering")
