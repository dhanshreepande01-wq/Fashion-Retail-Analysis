
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Fashion Trends Analysis",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Fashion Trends Analysis")
st.write("Explore brands, categories, products, discounts and price segments.")

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your Fashion Trends dataset",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload your Fashion Trends dataset.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Unable to read CSV file: {e}")
    st.stop()

# --------------------------------------------------
# CHECK REQUIRED COLUMNS
# --------------------------------------------------

required_columns = [
    "Original_Price", "Discount_Percentage", "Brand",
    "Category", "Gender", "Discount_Price",
    "Discounted_amount", "Price_Segment",
    "Product_Name", "Discount_Tier", "Product_ID"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(f"Missing columns: {missing_columns}")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

if df.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()

# --------------------------------------------------
# DATA PREPARATION
# --------------------------------------------------

data = df.copy()

numeric_columns = [
    "Original_Price",
    "Discount_Percentage",
    "Discount_Price",
    "Discounted_amount"
]

for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# --------------------------------------------------
# SIDEBAR SETTINGS - OUTLIERS
# --------------------------------------------------

st.sidebar.header("Settings")

remove_outliers = st.sidebar.checkbox(
    "Remove Outliers",
    value=False
)

def remove_iqr(frame, column):
    q1 = frame[column].quantile(0.25)
    q3 = frame[column].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return frame[
        frame[column].isna() |
        frame[column].between(lower, upper)
    ].copy()

if remove_outliers:
    data = remove_iqr(data, "Discount_Price")
    data = remove_iqr(data, "Original_Price")
    data = remove_iqr(data, "Discounted_amount")
    data = remove_iqr(data, "Discount_Percentage")


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def show_plot(fig):
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def bar_chart(series, title, xlabel, ylabel="Count"):
    if series.empty:
        st.info("No data available for this chart.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        x=series.index.astype(str),
        y=series.values,
        ax=ax
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)

    show_plot(fig)


def show_boxplot(column):
    values = data[column].dropna()

    if values.empty:
        st.info(f"No valid values available for {column}.")
        return

    fig, ax = plt.subplots(figsize=(9, 3))
    sns.boxplot(x=values, ax=ax)
    ax.set_title(f"Boxplot of {column}")

    show_plot(fig)


# --------------------------------------------------
# 1. DATA PREPARATION
# --------------------------------------------------

st.header("1. Data Preparation")

st.subheader("Dataset Preview")
st.dataframe(data.head(10), use_container_width=True)

col1, col2, col3 = st.columns(3)

col1.metric("Total Rows", data.shape[0])
col2.metric("Total Columns", data.shape[1])
col3.metric("Duplicate Rows", data.duplicated().sum())

st.subheader("Missing Values")

missing = df.isnull().sum().reset_index()
missing.columns = ["Column Name", "Missing Values"]

st.dataframe(missing, use_container_width=True)

st.subheader("Data Types")

data_types = data.dtypes.reset_index()
data_types.columns = ["Column Name", "Data Type"]

st.dataframe(data_types, use_container_width=True)

st.subheader("Column Names")
st.write(data.columns.tolist())


# --------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS
# --------------------------------------------------

st.header("2. Exploratory Data Analysis")

st.subheader("Descriptive Statistics")
st.dataframe(data.describe(include="all"), use_container_width=True)

st.metric("Total Records", data.shape[0])

col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Price Distribution")

    fig, ax = plt.subplots()
    sns.histplot(
        data=data,
        x="Original_Price",
        kde=True,
        ax=ax
    )
    ax.set_title("Original Price Distribution")
    show_plot(fig)

with col2:
    st.subheader("Discount Percentage Distribution")

    fig, ax = plt.subplots()
    sns.histplot(
        data=data,
        x="Discount_Percentage",
        kde=True,
        ax=ax
    )
    ax.set_title("Discount Percentage Distribution")
    show_plot(fig)

st.subheader("Top 10 Brands")

brand_count = data["Brand"].value_counts().head(10)
bar_chart(brand_count, "Top 10 Brands", "Brand")

st.subheader("Products by Category")

category_count = data["Category"].value_counts()
bar_chart(category_count, "Category Distribution", "Category")

st.subheader("Products by Gender")

gender_count = data["Gender"].value_counts()
bar_chart(gender_count, "Gender Distribution", "Gender")


# --------------------------------------------------
# 3. OUTLIER ANALYSIS
# --------------------------------------------------

st.header("3. Outlier Analysis")

outlier_columns = [
    "Discount_Price",
    "Original_Price",
    "Discounted_amount",
    "Discount_Percentage"
]

for col in outlier_columns:
    show_boxplot(col)

st.subheader("IQR Calculation")

selected_column = st.selectbox(
    "Select Column",
    outlier_columns
)

values = data[selected_column].dropna()

if values.empty:
    st.info("No valid numeric values available.")
else:
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    outliers = values[
        (values < lower_limit) |
        (values > upper_limit)
    ]

    col1, col2, col3 = st.columns(3)
    col1.metric("Q1", round(q1, 2))
    col2.metric("Q3", round(q3, 2))
    col3.metric("IQR", round(iqr, 2))

    col1, col2, col3 = st.columns(3)
    col1.metric("Lower Limit", round(lower_limit, 2))
    col2.metric("Upper Limit", round(upper_limit, 2))
    col3.metric("Outliers", len(outliers))


# --------------------------------------------------
# 4. BRAND AND CATEGORY ANALYSIS
# --------------------------------------------------

st.header("4. Brand & Category Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 10 Brands")
    st.dataframe(
        data["Brand"].value_counts().head(10)
    )

with col2:
    st.subheader("Category Analysis")
    st.dataframe(
        data["Category"].value_counts()
    )

st.subheader("Category vs Price Segment")

category_segment = pd.crosstab(
    data["Category"],
    data["Price_Segment"]
)

st.dataframe(category_segment, use_container_width=True)


# --------------------------------------------------
# 5. PRODUCT TYPE ANALYSIS
# --------------------------------------------------

st.header("5. Product Type Analysis")

st.subheader("Top 10 Products")

st.dataframe(
    data["Product_Name"].value_counts().head(10)
)

st.subheader("Products by Gender")

gender_products = data.groupby("Gender")["Product_Name"].count()

bar_chart(
    gender_products,
    "Products by Gender",
    "Gender"
)

st.subheader("Top 10 Categories by Product Count")

category_products = (
    data.groupby("Category")["Product_Name"]
    .count()
    .nlargest(10)
)

bar_chart(
    category_products,
    "Top 10 Categories by Product Count",
    "Category"
)


# --------------------------------------------------
# 6. DISCOUNT ANALYSIS
# --------------------------------------------------

st.header("6. Discount Analysis")

st.subheader("Products by Discount Tier")

discount_count = data["Discount_Tier"].value_counts()
st.dataframe(discount_count)

st.subheader("Average Discount by Product")

avg_discount = (
    data.groupby("Product_Name")["Discount_Percentage"]
    .mean()
    .nlargest(10)
)

st.dataframe(avg_discount)

st.subheader("Category vs Discount Tier")

discount_table = pd.crosstab(
    data["Category"],
    data["Discount_Tier"]
)

st.dataframe(discount_table, use_container_width=True)

if not discount_table.empty:
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.heatmap(
        discount_table,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        ax=ax
    )

    ax.set_title("Category vs Discount Tier")
    show_plot(fig)

st.subheader("Original Price vs Discount Percentage")

fig, ax = plt.subplots(figsize=(9, 5))

sns.scatterplot(
    data=data,
    x="Original_Price",
    y="Discount_Percentage",
    ax=ax
)

ax.set_title("Original Price vs Discount Percentage")
show_plot(fig)


# --------------------------------------------------
# 7. GENDER-BASED PREFERENCES
# --------------------------------------------------

st.header("7. Gender-Based Preferences")

st.subheader("Category Count by Gender")

gender_category = pd.crosstab(
    data["Gender"],
    data["Category"]
)

st.dataframe(gender_category, use_container_width=True)

fig, ax = plt.subplots(figsize=(10, 5))

sns.countplot(
    data=data,
    x="Category",
    hue="Gender",
    ax=ax
)

ax.set_title("Category Preferences by Gender")
ax.tick_params(axis="x", rotation=45)
show_plot(fig)

st.subheader("Average Discount by Gender")

avg_gender = data.groupby("Gender")["Discount_Percentage"].mean()

st.dataframe(avg_gender)

fig, ax = plt.subplots(figsize=(8, 5))

sns.boxplot(
    data=data,
    x="Gender",
    y="Discount_Percentage",
    ax=ax
)

ax.set_title("Discount Percentage by Gender")
show_plot(fig)


# --------------------------------------------------
# 8. PRICE SEGMENT ANALYSIS
# --------------------------------------------------

st.header("8. Price Segment Analysis")

st.subheader("Price Segment Count")

price_count = data["Price_Segment"].value_counts()
st.dataframe(price_count)

st.subheader("Category vs Price Segment")

price_table = pd.crosstab(
    data["Category"],
    data["Price_Segment"],
    margins=True
)

st.dataframe(price_table, use_container_width=True)

st.subheader("Brand vs Price Segment")

brand_price = pd.pivot_table(
    data,
    index="Brand",
    columns="Price_Segment",
    values="Original_Price",
    aggfunc="mean"
)

st.dataframe(brand_price, use_container_width=True)

if not brand_price.empty:
    fig, ax = plt.subplots(figsize=(12, 7))

    sns.heatmap(
        brand_price,
        annot=False,
        cmap="YlGnBu",
        ax=ax
    )

    ax.set_title("Average Original Price by Brand and Price Segment")
    show_plot(fig)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.success("Fashion Trends Analysis Completed!")

st.caption(
    "Developed using Python, Pandas, Seaborn, "
    "Matplotlib and Streamlit."
)
