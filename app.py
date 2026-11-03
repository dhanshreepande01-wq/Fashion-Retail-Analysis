
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Fashion Trends Analysis",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Fashion Trends Analysis")
st.write("Explore brands, categories, products, discounts and price segments.")

# File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Please upload your Fashion Trends dataset.")
    st.stop()

df = pd.read_csv(uploaded_file)

# Check required columns
required_columns = [
    "Original_Price", "Discount_Percentage", "Brand",
    "Category", "Gender", "Discount_Price",
    "Discounted_amount", "Price_Segment",
    "Product_Name", "Discount_Tier", "Product_ID"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    st.error(f"Missing columns: {missing}")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

# Outlier Removal
st.sidebar.header("Settings")

remove_outliers = st.sidebar.checkbox(
    "Remove Outliers",
    value=False
)

data = df.copy()

def remove_iqr(frame, column):
    q1 = frame[column].quantile(0.25)
    q3 = frame[column].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return frame[
        (frame[column] >= lower) &
        (frame[column] <= upper)
    ].copy()

if remove_outliers:
    data = remove_iqr(data, "Discount_Price")
    data = remove_iqr(data, "Original_Price")

# Helper Function
def show_plot(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def bar_chart(series, title, xlabel):
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        x=series.index.astype(str),
        y=series.values,
        ax=ax
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    show_plot(fig)


# 1. DATA PREPARATION
st.header("1. Data Preparation")

st.subheader("Dataset Preview")
st.dataframe(data.head(10), use_container_width=True)

col1, col2, col3 = st.columns(3)

col1.metric("Total Rows", data.shape[0])
col2.metric("Total Columns", data.shape[1])
col3.metric("Duplicate Rows", data.duplicated().sum())

st.subheader("Missing Values")

missing = df.isnull().sum().reset_index()
missing.columns = ['Column Name', 'Missing Values']

st.dataframe(missing, use_container_width=True)

st.subheader("Data Types")

data_types = df.dtypes.reset_index()
data_types.columns = ['Column Name', 'Data Type']

st.dataframe(data_types, use_container_width=True)

st.subheader("Column Names")
st.write(data.columns.tolist())


# 2. EXPLORATORY DATA ANALYSIS
st.header("2. Exploratory Data Analysis")

st.subheader("Descriptive Statistics")
st.dataframe(data.describe(include="all"))

st.metric("Total Records", data.shape[0])

col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Price Distribution")

    fig, ax = plt.subplots()
    sns.histplot(data=data, x="Original_Price", kde=True, ax=ax)
    ax.set_title("Original Price Distribution")
    show_plot(fig)

with col2:
    st.subheader("Discount Percentage Distribution")

    fig, ax = plt.subplots()
    sns.histplot(data=data, x="Discount_Percentage", kde=True, ax=ax)
    ax.set_title("Discount Percentage Distribution")
    show_plot(fig)

# Top 10 Brands
st.subheader("Top 10 Brands")

b = data["Brand"].value_counts().nlargest(10)
bar_chart(b, "Top 10 Brands", "Brand")

# Category
st.subheader("Products by Category")

c = data["Category"].value_counts()
bar_chart(c, "Category Distribution", "Category")

# Gender
st.subheader("Products by Gender")

g = data["Gender"].value_counts()
bar_chart(g, "Gender Distribution", "Gender")


# OUTLIER ANALYSIS
st.header("3. Outlier Analysis")

for col in [
    "Discount_Price",
    "Original_Price",
    "Discounted_amount",
    "Discount_Percentage"
]:
    fig, ax = plt.subplots(figsize=(9, 3))
    sns.boxplot(data=data, x=col, ax=ax)
    ax.set_title(f"Boxplot of {col}")
    show_plot(fig)

st.subheader("IQR Calculation")

p = st.selectbox(
    "Select Column",
    ["Discount_Price", "Original_Price",
     "Discounted_amount", "Discount_Percentage"]
)

q1 = data[p].quantile(0.25)
q3 = data[p].quantile(0.75)
iqr = q3 - q1

c1 = q1 - 1.5 * iqr
c2 = q3 + 1.5 * iqr

outliers = data[(data[p] < c1) | (data[p] > c2)]

col1, col2, col3 = st.columns(3)
col1.metric("Q1", round(q1, 2))
col2.metric("Q3", round(q3, 2))
col3.metric("IQR", round(iqr, 2))

col1, col2, col3 = st.columns(3)
col1.metric("Lower Limit", round(c1, 2))
col2.metric("Upper Limit", round(c2, 2))
col3.metric("Outliers", len(outliers))


# 4. BRAND AND CATEGORY ANALYSIS
st.header("4. Brand & Category Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 10 Brands")
    st.dataframe(
        data["Brand"].value_counts().nlargest(10)
    )

with col2:
    st.subheader("Category Analysis")
    st.dataframe(
        data["Category"].value_counts()
    )

st.subheader("Category vs Price Segment")

u = data.groupby("Price_Segment")["Category"].size()
st.dataframe(u)

category_segment = pd.crosstab(
    data["Category"],
    data["Price_Segment"]
)

st.dataframe(category_segment)


# 5. PRODUCT TYPE ANALYSIS
st.header("5. Product Type Analysis")

st.subheader("Top 10 Products")

st.dataframe(
    data["Product_Name"].value_counts().nlargest(10)
)

st.subheader("Products by Gender")

g = data.groupby("Gender")["Product_Name"].count()

bar_chart(g, "Products by Gender", "Gender")

st.subheader("Top 10 Categories by Product Count")

category_products = (
    data.groupby("Category")["Product_Name"]
    .count()
    .sort_values(ascending=False)
    .head(10)
)

bar_chart(
    category_products,
    "Top 10 Categories by Product Count",
    "Category"
)


# 6. DISCOUNT ANALYSIS
st.header("6. Discount Analysis")

st.subheader("Products by Discount Tier")

st.dataframe(
    data.groupby("Discount_Tier").size()
)

st.subheader("Average Discount by Product")

avg_discount = (
    data.groupby("Product_Name")["Discount_Percentage"]
    .mean()
    .nlargest(10)
)

st.dataframe(avg_discount)

st.subheader("Category vs Discount Tier")

pi = pd.pivot_table(
    data=data,
    index="Category",
    columns="Discount_Tier",
    values="Product_ID",
    aggfunc="count",
    fill_value=0
)

st.dataframe(pi)

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(pi, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
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


# 7. GENDER-BASED PREFERENCES
st.header("7. Gender-Based Preferences")

st.subheader("Category Count by Gender")

st.dataframe(
    data.groupby("Gender")["Category"].count()
)

fig, ax = plt.subplots(figsize=(10, 5))

sns.countplot(
    data=data,
    x="Category",
    hue="Gender",
    ax=ax
)

ax.set_title("Category Preferences by Gender")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
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


# 8. PRICE SEGMENT ANALYSIS
st.header("8. Price Segment Analysis")

st.subheader("Price Segment Count")

st.dataframe(
    data["Price_Segment"].value_counts()
)

st.subheader("Category vs Price Segment")

price_table = pd.crosstab(
    index=data["Category"],
    columns=data["Price_Segment"],
    margins=True
)

st.dataframe(price_table)

st.subheader("Brand vs Price Segment")

a = pd.pivot_table(
    data,
    index="Brand",
    columns="Price_Segment",
    values="Original_Price",
    aggfunc="mean"
)

st.dataframe(a)

fig, ax = plt.subplots(figsize=(12, 7))

sns.heatmap(
    a,
    annot=False,
    fmt=".0f",
    cmap="YlGnBu",
    ax=ax
)

ax.set_title("Average Original Price by Brand and Price Segment")
fig.tight_layout()
show_plot(fig)


# FOOTER
st.divider()

st.success("Fashion Trends Analysis Completed!")

st.caption(
    "Developed using Python, Pandas, Seaborn, "
    "Matplotlib and Streamlit."
)
