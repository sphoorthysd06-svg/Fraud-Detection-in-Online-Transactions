# ============================================================
# FRAUD DETECTION IN ONLINE TRANSACTIONS
# Fin-Care Starpowerz Digital Technology Pvt. Ltd.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fraud Detection Analytics",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================================================
# FIN-CARE OFFICE BACKGROUND
# ============================================================

import base64

def set_background(image_file):

    with open(image_file, "rb") as file:
        image_data = base64.b64encode(
            file.read()
        ).decode()

    st.markdown(
        f"""
        <style>

        .stApp {{
            background-image:
                linear-gradient(
                    rgba(255, 255, 255, 0.88),
                    rgba(255, 255, 255, 0.88)
                ),
                url("data:image/png;base64,{image_data}");

            background-size: cover;
            background-position: center top;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

set_background("Fin-Care Office.png")

# ============================================================
# LOGIN DETAILS
# ============================================================

USERNAME = "sony"
PASSWORD = "Sony@4961"


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "Home"


# ============================================================
# LOGIN PAGE
# ============================================================

if not st.session_state.logged_in:

    # --------------------------------------------------------
    # CENTERED HEADER
    # --------------------------------------------------------

    left, header, right = st.columns([1, 2, 1])

    with header:

        st.subheader(
            "🛡️ Fraud Detection in Online Transactions"
        )

        st.caption(
            "Fin-Care Starpowerz Digital Technology Pvt. Ltd."
        )

    st.divider()


    # --------------------------------------------------------
    # CENTERED LOGIN
    # --------------------------------------------------------

    left, login_box, right = st.columns([1, 2, 1])

    with login_box:

        st.subheader("🔐 Login")

        user_id = st.text_input(
            "👤 User ID",
            placeholder="Enter User ID"
        )

        password = st.text_input(
            "🔑 Password",
            type="password",
            placeholder="Enter Password"
        )

        login_button = st.button(
            "🔐 Login",
            use_container_width=True
        )


        if login_button:

            if user_id == "" or password == "":

                st.warning(
                    "⚠️ Please enter User ID and Password."
                )

            elif (
                user_id == USERNAME
                and password == PASSWORD
            ):

                st.session_state.logged_in = True
                st.session_state.page = "Home"

                st.rerun()

            else:

                st.error(
                    "❌ Invalid User ID or Password."
                )


    # Stop here until login
    st.stop()


# ============================================================
# LOAD FRAUD DETECTION DATASET
# ============================================================

try:

    df = pd.read_csv(
        "Fraud_Detection_Cleaned.csv"
    )

except FileNotFoundError:

    st.error(
        "❌ Fraud_Detection_Cleaned.csv was not found."
    )

    st.info(
        "Please keep Fraud_Detection_Cleaned.csv "
        "in the same folder as Stream.py."
    )

    st.stop()


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)


# ============================================================
# CONVERT DATE
# ============================================================

if "Transaction_Date" in df.columns:

    df["Transaction_Date"] = pd.to_datetime(
        df["Transaction_Date"],
        errors="coerce"
    )


# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [

    "Transaction_Amount",

    "IP_Risk_Score",

    "Device_Risk_Score",

    "Velocity_Score",

    "Transactions_Last_24H",

    "Failed_Transactions_24H",

    "Previous_Transactions",

    "Account_Age_Months"
]


for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ============================================================
# FRAUD FLAG
# ============================================================

if "Fraud_Flag" not in df.columns:

    st.error(
        "❌ Fraud_Flag column is missing from the dataset."
    )

    st.stop()


if df["Fraud_Flag"].dtype == "object":

    df["Fraud_Flag"] = (

        df["Fraud_Flag"]
        .astype(str)
        .str.strip()
        .str.lower()

        .map({

            "yes": 1,

            "true": 1,

            "fraud": 1,

            "fraudulent": 1,

            "1": 1,

            "no": 0,

            "false": 0,

            "legitimate": 0,

            "0": 0

        })

    )


df["Fraud_Flag"] = pd.to_numeric(
    df["Fraud_Flag"],
    errors="coerce"
).fillna(0).astype(int)


# ============================================================
# BASIC KPI CALCULATIONS
# ============================================================

total_transactions = len(df)


fraud_transactions = int(
    (df["Fraud_Flag"] == 1).sum()
)


legitimate_transactions = (
    total_transactions
    - fraud_transactions
)


fraud_rate = (

    fraud_transactions
    / total_transactions

    if total_transactions > 0

    else 0
)


if "Transaction_Amount" in df.columns:

    total_amount = df[
        "Transaction_Amount"
    ].sum()

else:

    total_amount = 0


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛡️ Fraud Detection")

    st.write(
        "Fin-Care Starpowerz Digital Technology Pvt. Ltd."
    )

    st.divider()


    # --------------------------------------------------------
    # HOME
    # --------------------------------------------------------

    if st.button(
        "🏠 Home",
        use_container_width=True
    ):

        st.session_state.page = "Home"

        st.rerun()


    # --------------------------------------------------------
    # DASHBOARD
    # --------------------------------------------------------

    if st.button(
        "📊 Dashboard",
        use_container_width=True
    ):

        st.session_state.page = "Dashboard"

        st.rerun()


    # --------------------------------------------------------
    # FRAUD ANALYTICS
    # --------------------------------------------------------

    if st.button(
        "📈 Fraud Analytics",
        use_container_width=True
    ):

        st.session_state.page = "Fraud Analytics"

        st.rerun()


    # --------------------------------------------------------
    # RISK ANALYSIS
    # --------------------------------------------------------

    if st.button(
        "⚠️ Risk Analysis",
        use_container_width=True
    ):

        st.session_state.page = "Risk Analysis"

        st.rerun()


    # --------------------------------------------------------
    # TRANSACTIONS
    # --------------------------------------------------------

    if st.button(
        "💳 Transactions",
        use_container_width=True
    ):

        st.session_state.page = "Transactions"

        st.rerun()


    # --------------------------------------------------------
    # FRAUD PREDICTION
    # --------------------------------------------------------

    if st.button(
        "🔮 Fraud Prediction",
        use_container_width=True
    ):

        st.session_state.page = "Fraud Prediction"

        st.rerun()


    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    if st.button(
        "📋 Model Performance",
        use_container_width=True
    ):

        st.session_state.page = "Model Performance"

        st.rerun()


    st.divider()


    # --------------------------------------------------------
    # LOGOUT - LAST BUTTON
    # --------------------------------------------------------

    if st.button(
        "🚪 Logout",
        use_container_width=True
    ):

        st.session_state.logged_in = False

        st.session_state.page = "Home"

        st.rerun()


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "Home":

    st.title(
        "🛡️ Welcome to Fraud & Risk Management Analytics"
    )

    st.write(
        "This Streamlit application analyzes online "
        "transaction data and presents fraud-related "
        "patterns, risk indicators, transaction behaviour "
        "and machine-learning results."
    )

    st.divider()


    col1, col2, col3 = st.columns(3)


    with col1:

        st.info(
            "📊 Dashboard\n\n"
            "View fraud KPIs, transaction trends, "
            "fraud categories and risk indicators."
        )


    with col2:

        st.warning(
            "⚠️ Risk Analysis\n\n"
            "Analyze risk scores, location mismatch "
            "and transaction risk indicators."
        )


    with col3:

        st.success(
            "🔮 Fraud Prediction\n\n"
            "Use machine learning to predict "
            "whether a transaction is fraudulent."
        )


    st.divider()


    st.subheader("Project Information")


    project_data = pd.DataFrame({

        "Item": [

            "Project",

            "Company",

            "Domain",

            "Dataset",

            "Total Records",

            "Fraudulent Records"

        ],

        "Details": [

            "Fraud Detection in Online Transactions",

            "Fin-Care Starpowerz Digital Technology Pvt. Ltd.",

            "Financial Services / Fraud Risk Management",

            "Fraud_Detection_Cleaned.csv",

            f"{total_transactions:,}",

            f"{fraud_transactions:,}"

        ]

    })


    st.dataframe(
        project_data,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DASHBOARD
# ============================================================

elif st.session_state.page == "Dashboard":

    st.title(
        "🛡️ Fraud Detection in Online Transactions"
    )

    st.caption(
        "Fin-Care Starpowerz Digital Technology Pvt. Ltd."
    )

    st.caption(
        "Fraud & Risk Management Analytics Dashboard"
    )

    st.divider()


    # ========================================================
    # KPI SECTION
    # ========================================================

    k1, k2, k3, k4, k5 = st.columns(5)


    with k1:

        st.metric(
            "Total Transactions",
            f"{total_transactions / 1000:.0f}K"
        )


    with k2:

        st.metric(
            "Fraudulent Transactions",
            f"{fraud_transactions}"
        )


    with k3:

        st.metric(
            "Legitimate Transactions",
            f"{legitimate_transactions / 1000:.0f}K"
        )


    with k4:

        st.metric(
            "Fraud Rate",
            f"{fraud_rate:.2f}"
        )


    with k5:

        st.metric(
            "Total Transaction Amount",
            f"{total_amount / 1000000:.2f}M"
        )


    st.divider()


    # ========================================================
    # ROW 1
    # ========================================================

    col1, col2, col3 = st.columns(3)


    # --------------------------------------------------------
    # FRAUD VS LEGITIMATE
    # --------------------------------------------------------

    with col1:

        status_data = pd.DataFrame({

            "Status": [

                "Legitimate",

                "Fraudulent"

            ],

            "Transactions": [

                legitimate_transactions,

                fraud_transactions

            ]

        })


        fig = px.pie(

            status_data,

            names="Status",

            values="Transactions",

            hole=0.60,

            title="Fraudulent vs Legitimate Transactions"

        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # FRAUD BY TRANSACTION TYPE
    # --------------------------------------------------------

    with col2:

        if "Transaction_Type" in df.columns:

            transaction_type = (

                df[df["Fraud_Flag"] == 1]

                .groupby(
                    "Transaction_Type"
                )

                .size()

                .reset_index(
                    name="Fraudulent Transactions"
                )

                .sort_values(
                    "Fraudulent Transactions",
                    ascending=False
                )

            )


            fig = px.bar(

                transaction_type,

                x="Transaction_Type",

                y="Fraudulent Transactions",

                text="Fraudulent Transactions",

                title="Fraudulent Transactions by Transaction Type"

            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # --------------------------------------------------------
    # FRAUD TREND BY YEAR
    # --------------------------------------------------------

    with col3:

        if "Transaction_Date" in df.columns:

            yearly = (

                df[
                    df["Fraud_Flag"] == 1
                ]

                .assign(
                    Year=df["Transaction_Date"].dt.year
                )

                .groupby("Year")

                .size()

                .reset_index(
                    name="Fraudulent Transactions"
                )

            )

        elif "Year" in df.columns:

            yearly = (

                df[
                    df["Fraud_Flag"] == 1
                ]

                .groupby("Year")

                .size()

                .reset_index(
                    name="Fraudulent Transactions"
                )

            )

        else:

            yearly = pd.DataFrame()


        if not yearly.empty:

            fig = px.line(

                yearly,

                x="Year",

                y="Fraudulent Transactions",

                markers=True,

                text="Fraudulent Transactions",

                title="Fraudulent Transactions Trend by Year"

            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # ========================================================
    # ROW 2
    # ========================================================

    col4, col5, col6 = st.columns(3)


    # --------------------------------------------------------
    # MERCHANT CATEGORY
    # --------------------------------------------------------

    with col4:

        if "Merchant_Category" in df.columns:

            merchant_data = (

                df[
                    df["Fraud_Flag"] == 1
                ]

                .groupby(
                    "Merchant_Category"
                )

                .size()

                .reset_index(
                    name="Fraudulent Transactions"
                )

                .sort_values(
                    "Fraudulent Transactions",
                    ascending=True
                )

            )


            fig = px.bar(

                merchant_data,

                x="Fraudulent Transactions",

                y="Merchant_Category",

                orientation="h",

                text="Fraudulent Transactions",

                title="Fraudulent Transactions by Merchant Category"

            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # --------------------------------------------------------
    # CUSTOMER TYPE
    # --------------------------------------------------------

    with col5:

        if "Customer_Type" in df.columns:

            customer_data = (

                df[
                    df["Fraud_Flag"] == 1
                ]

                .groupby(
                    "Customer_Type"
                )

                .size()

                .reset_index(
                    name="Fraudulent Transactions"
                )

                .sort_values(
                    "Fraudulent Transactions",
                    ascending=False
                )

            )


            fig = px.bar(

                customer_data,

                x="Customer_Type",

                y="Fraudulent Transactions",

                text="Fraudulent Transactions",

                title="Fraudulent Transactions by Customer Type"

            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # --------------------------------------------------------
    # DEVICE TYPE
    # --------------------------------------------------------

    with col6:

        if "Device_Type" in df.columns:

            device_data = (

                df[
                    df["Fraud_Flag"] == 1
                ]

                .groupby(
                    "Device_Type"
                )

                .size()

                .reset_index(
                    name="Fraudulent Transactions"
                )

                .sort_values(
                    "Fraudulent Transactions",
                    ascending=False
                )

            )


            fig = px.bar(

                device_data,

                x="Device_Type",

                y="Fraudulent Transactions",

                text="Fraudulent Transactions",

                title="Fraudulent Transactions by Device Type"

            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # ========================================================
    # ROW 3
    # ========================================================

    col7, col8, col9 = st.columns(3)


    # --------------------------------------------------------
    # INTERNATIONAL TRANSACTIONS
    # --------------------------------------------------------

    with col7:

        if "International_Transaction" in df.columns:

            international_data = (

                df[
                    "International_Transaction"
                ]

                .astype(str)

                .str.strip()

                .str.title()

                .value_counts()

                .reset_index()

            )


            international_data.columns = [

                "International Transaction",

                "Transactions"

            ]


            fig = px.pie(

                international_data,

                names="International Transaction",

                values="Transactions",

                hole=0.55,

                title="Total Transaction Analysis"

            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # --------------------------------------------------------
    # LOCATION MISMATCH
    # --------------------------------------------------------

    with col8:

        if "Location_Mismatch" in df.columns:

            mismatch_data = (

                df[
                    "Location_Mismatch"
                ]

                .astype(str)

                .str.strip()

                .str.title()

                .value_counts()

                .reset_index()

            )


            mismatch_data.columns = [

                "Location Mismatch",

                "Transactions"

            ]


            fig = px.pie(

                mismatch_data,

                names="Location Mismatch",

                values="Transactions",

                hole=0.55,

                title="Location Mismatch Fraud Analysis"

            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # --------------------------------------------------------
    # AVERAGE RISK SCORE
    # --------------------------------------------------------

    with col9:

        risk_columns = [

            "IP_Risk_Score",

            "Device_Risk_Score",

            "Velocity_Score"

        ]


        available_risk = [

            column

            for column in risk_columns

            if column in df.columns

        ]


        if len(available_risk) > 0:

            risk_data = (

                df

                .groupby(
                    "Fraud_Flag"
                )

                [available_risk]

                .mean()

                .reset_index()

            )


            risk_data["Fraud Status"] = (

                risk_data["Fraud_Flag"]

                .map({

                    0: "Legitimate",

                    1: "Fraudulent"

                })

            )


            risk_long = risk_data.melt(

                id_vars=[

                    "Fraud_Flag",

                    "Fraud Status"

                ],

                value_vars=available_risk,

                var_name="Risk Type",

                value_name="Average Score"

            )


            risk_long["Risk Type"] = (

                risk_long["Risk Type"]

                .str.replace(
                    "_",
                    " ",
                    regex=False
                )

            )


            fig = px.bar(

                risk_long,

                x="Fraud Status",

                y="Average Score",

                color="Risk Type",

                barmode="group",

                text_auto=".0f",

                title="Average Risk Score by Fraud Status"

            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# FRAUD ANALYTICS
# ============================================================

elif st.session_state.page == "Fraud Analytics":

    st.title("📈 Fraud Analytics")

    st.write(
        "Detailed analysis of fraudulent transactions."
    )

    st.divider()


    fraud_df = df[
        df["Fraud_Flag"] == 1
    ]


    col1, col2 = st.columns(2)


    with col1:

        if "Transaction_Type" in df.columns:

            data = (

                fraud_df

                .groupby(
                    "Transaction_Type"
                )

                .size()

                .reset_index(
                    name="Fraudulent Transactions"
                )

            )


            fig = px.bar(

                data,

                x="Transaction_Type",

                y="Fraudulent Transactions",

                text="Fraudulent Transactions",

                title="Fraud by Transaction Type"

            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


    with col2:

        if "Channel" in df.columns:

            data = (

                fraud_df

                .groupby(
                    "Channel"
                )

                .size()

                .reset_index(
                    name="Fraudulent Transactions"
                )

            )


            fig = px.bar(

                data,

                x="Channel",

                y="Fraudulent Transactions",

                text="Fraudulent Transactions",

                title="Fraud by Channel"

            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


    col3, col4 = st.columns(2)


    with col3:

        if "Merchant_Category" in df.columns:

            data = (

                fraud_df

                .groupby(
                    "Merchant_Category"
                )

                .size()

                .reset_index(
                    name="Fraudulent Transactions"
                )

            )


            fig = px.pie(

                data,

                names="Merchant_Category",

                values="Fraudulent Transactions",

                title="Fraud by Merchant Category"

            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


    with col4:

        if "Device_Type" in df.columns:

            data = (

                fraud_df

                .groupby(
                    "Device_Type"
                )

                .size()

                .reset_index(
                    name="Fraudulent Transactions"
                )

            )


            fig = px.pie(

                data,

                names="Device_Type",

                values="Fraudulent Transactions",

                hole=0.5,

                title="Fraud by Device Type"

            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# RISK ANALYSIS
# ============================================================

elif st.session_state.page == "Risk Analysis":

    st.title("⚠️ Risk Analysis")

    st.write(
        "Analysis of transaction risk indicators."
    )

    st.divider()


    col1, col2, col3 = st.columns(3)


    with col1:

        if "IP_Risk_Score" in df.columns:

            st.metric(

                "Average IP Risk Score",

                f"{df['IP_Risk_Score'].mean():.2f}"

            )


    with col2:

        if "Device_Risk_Score" in df.columns:

            st.metric(

                "Average Device Risk Score",

                f"{df['Device_Risk_Score'].mean():.2f}"

            )


    with col3:

        if "Velocity_Score" in df.columns:

            st.metric(

                "Average Velocity Score",

                f"{df['Velocity_Score'].mean():.2f}"

            )


    st.divider()


    risk_columns = [

        "IP_Risk_Score",

        "Device_Risk_Score",

        "Velocity_Score"

    ]


    available_risk = [

        column

        for column in risk_columns

        if column in df.columns

    ]


    if available_risk:

        risk_data = (

            df

            .groupby(
                "Fraud_Flag"
            )

            [available_risk]

            .mean()

            .reset_index()

        )


        risk_data["Fraud Status"] = (

            risk_data["Fraud_Flag"]

            .map({

                0: "Legitimate",

                1: "Fraudulent"

            })

        )


        risk_long = risk_data.melt(

            id_vars=[

                "Fraud_Flag",

                "Fraud Status"

            ],

            value_vars=available_risk,

            var_name="Risk Type",

            value_name="Average Score"

        )


        fig = px.bar(

            risk_long,

            x="Risk Type",

            y="Average Score",

            color="Fraud Status",

            barmode="group",

            title="Risk Score Comparison"

        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    if "Location_Mismatch" in df.columns:

        mismatch = (

            df

            .groupby([

                "Location_Mismatch",

                "Fraud_Flag"

            ])

            .size()

            .reset_index(
                name="Transactions"
            )

        )


        mismatch["Fraud Status"] = (

            mismatch["Fraud_Flag"]

            .map({

                0: "Legitimate",

                1: "Fraudulent"

            })

        )


        fig = px.bar(

            mismatch,

            x="Location_Mismatch",

            y="Transactions",

            color="Fraud Status",

            barmode="group",

            title="Location Mismatch Analysis"

        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# TRANSACTIONS
# ============================================================

elif st.session_state.page == "Transactions":

    st.title("💳 Transactions")

    st.write(
        "Search and filter transaction records."
    )

    st.divider()


    col1, col2, col3 = st.columns(3)


    with col1:

        status_filter = st.selectbox(

            "Transaction Status",

            [

                "All",

                "Fraudulent",

                "Legitimate"

            ]

        )


    with col2:

        if "Transaction_Type" in df.columns:

            transaction_types = [

                "All"

            ] + sorted(

                df["Transaction_Type"]

                .dropna()

                .astype(str)

                .unique()

                .tolist()

            )


            type_filter = st.selectbox(

                "Transaction Type",

                transaction_types

            )

        else:

            type_filter = "All"


    with col3:

        if "Device_Type" in df.columns:

            device_types = [

                "All"

            ] + sorted(

                df["Device_Type"]

                .dropna()

                .astype(str)

                .unique()

                .tolist()

            )


            device_filter = st.selectbox(

                "Device Type",

                device_types

            )

        else:

            device_filter = "All"


    filtered_df = df.copy()


    if status_filter == "Fraudulent":

        filtered_df = filtered_df[

            filtered_df["Fraud_Flag"] == 1

        ]


    elif status_filter == "Legitimate":

        filtered_df = filtered_df[

            filtered_df["Fraud_Flag"] == 0

        ]


    if (

        type_filter != "All"

        and "Transaction_Type" in filtered_df.columns

    ):

        filtered_df = filtered_df[

            filtered_df["Transaction_Type"]

            .astype(str)

            == type_filter

        ]


    if (

        device_filter != "All"

        and "Device_Type" in filtered_df.columns

    ):

        filtered_df = filtered_df[

            filtered_df["Device_Type"]

            .astype(str)

            == device_filter

        ]


    st.write(

        f"Showing {len(filtered_df):,} transactions"

    )


    st.dataframe(

        filtered_df,

        use_container_width=True,

        height=500

    )


# ============================================================
# FRAUD PREDICTION
# ============================================================

elif st.session_state.page == "Fraud Prediction":

    st.title("🔮 Fraud Prediction")

    st.write(
        "Train a Random Forest model and predict a transaction."
    )

    st.divider()


    target = "Fraud_Flag"


    remove_columns = [

        target,

        "Fraud_Status",

        "Review_Status",

        "Transaction_ID",

        "Customer_ID",

        "Transaction_Date",

        "Transaction_Time"

    ]


    feature_columns = [

        column

        for column in df.columns

        if column not in remove_columns

    ]


    X = df[
        feature_columns
    ].copy()


    y = df[
        target
    ].copy()


    categorical_features = (

        X.select_dtypes(
            include=["object"]
        )
        .columns
        .tolist()

    )


    numeric_features = (

        X.select_dtypes(
            exclude=["object"]
        )
        .columns
        .tolist()

    )


    numeric_transformer = Pipeline(

        steps=[

            (
                "imputer",

                SimpleImputer(
                    strategy="median"
                )

            )

        ]

    )


    categorical_transformer = Pipeline(

        steps=[

            (
                "imputer",

                SimpleImputer(
                    strategy="most_frequent"
                )

            ),

            (
                "onehot",

                OneHotEncoder(
                    handle_unknown="ignore"
                )

            )

        ]

    )


    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numeric",

                numeric_transformer,

                numeric_features

            ),

            (
                "categorical",

                categorical_transformer,

                categorical_features

            )

        ]

    )


    model = Pipeline(

        steps=[

            (
                "preprocessor",

                preprocessor

            ),

            (
                "model",

                RandomForestClassifier(

                    n_estimators=100,

                    random_state=42,

                    class_weight="balanced"

                )

            )

        ]

    )


    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )


    with st.spinner(
        "Training Random Forest model..."
    ):

        model.fit(
            X_train,
            y_train
        )


    st.success(
        "Model trained successfully."
    )


    st.subheader(
        "Enter Transaction Details"
    )


    input_data = {}


    input_columns = st.columns(3)


    for i, column in enumerate(feature_columns):

        with input_columns[i % 3]:

            if column in categorical_features:

                values = (

                    df[column]

                    .dropna()

                    .astype(str)

                    .unique()

                    .tolist()

                )


                if len(values) > 0:

                    input_data[column] = st.selectbox(

                        column,

                        values,

                        key="prediction_" + column

                    )

                else:

                    input_data[column] = ""


            else:

                median_value = pd.to_numeric(

                    df[column],

                    errors="coerce"

                ).median()


                if pd.isna(median_value):

                    median_value = 0


                input_data[column] = st.number_input(

                    column,

                    value=float(median_value),

                    key="prediction_" + column

                )


    st.divider()


    if st.button(

        "🔍 Predict Transaction",

        use_container_width=True

    ):

        input_df = pd.DataFrame(

            [input_data]

        )


        prediction = model.predict(

            input_df

        )[0]


        probability = model.predict_proba(

            input_df

        )[0][1]


        if prediction == 1:

            st.error(

                f"🚨 FRAUDULENT TRANSACTION\n\n"
                f"Fraud Probability: {probability:.2%}"

            )

        else:

            st.success(

                f"✅ LEGITIMATE TRANSACTION\n\n"
                f"Fraud Probability: {probability:.2%}"

            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif st.session_state.page == "Model Performance":

    st.title("📋 Model Performance")

    st.write(
        "Comparison of machine-learning classification models."
    )

    st.divider()


    target = "Fraud_Flag"


    remove_columns = [

        target,

        "Fraud_Status",

        "Review_Status",

        "Transaction_ID",

        "Customer_ID",

        "Transaction_Date",

        "Transaction_Time"

    ]


    feature_columns = [

        column

        for column in df.columns

        if column not in remove_columns

    ]


    X = df[
        feature_columns
    ].copy()


    y = df[
        target
    ].copy()


    categorical_features = (

        X.select_dtypes(
            include=["object"]
        )
        .columns
        .tolist()

    )


    numeric_features = (

        X.select_dtypes(
            exclude=["object"]
        )
        .columns
        .tolist()

    )


    numeric_transformer = Pipeline(

        steps=[

            (
                "imputer",

                SimpleImputer(
                    strategy="median"
                )

            )

        ]

    )


    categorical_transformer = Pipeline(

        steps=[

            (
                "imputer",

                SimpleImputer(
                    strategy="most_frequent"
                )

            ),

            (
                "onehot",

                OneHotEncoder(
                    handle_unknown="ignore"
                )

            )

        ]

    )


    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numeric",

                numeric_transformer,

                numeric_features

            ),

            (
                "categorical",

                categorical_transformer,

                categorical_features

            )

        ]

    )


    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )


    models = {

        "Logistic Regression":

            LogisticRegression(

                max_iter=1000,

                class_weight="balanced"

            ),

        "Decision Tree":

            DecisionTreeClassifier(

                random_state=42,

                class_weight="balanced"

            ),

        "Random Forest":

            RandomForestClassifier(

                n_estimators=100,

                random_state=42,

                class_weight="balanced"

            )

    }


    results = []


    with st.spinner(
        "Training models..."
    ):

        for model_name, classifier in models.items():

            pipeline = Pipeline(

                steps=[

                    (
                        "preprocessor",

                        preprocessor

                    ),

                    (
                        "classifier",

                        classifier

                    )

                ]

            )


            pipeline.fit(

                X_train,

                y_train

            )


            predictions = pipeline.predict(

                X_test

            )


            probabilities = pipeline.predict_proba(

                X_test

            )[:, 1]


            results.append({

                "Model":

                    model_name,

                "Accuracy":

                    accuracy_score(

                        y_test,

                        predictions

                    ),

                "Precision":

                    precision_score(

                        y_test,

                        predictions,

                        zero_division=0

                    ),

                "Recall":

                    recall_score(

                        y_test,

                        predictions,

                        zero_division=0

                    ),

                "F1-Score":

                    f1_score(

                        y_test,

                        predictions,

                        zero_division=0

                    ),

                "ROC-AUC":

                    roc_auc_score(

                        y_test,

                        probabilities

                    )

            })


    results_df = pd.DataFrame(
        results
    )


    st.subheader(
        "Model Comparison"
    )


    display_results = results_df.copy()


    metric_columns = [

        "Accuracy",

        "Precision",

        "Recall",

        "F1-Score",

        "ROC-AUC"

    ]


    for column in metric_columns:

        display_results[column] = (

            display_results[column]
            .round(3)

        )


    st.dataframe(

        display_results,

        use_container_width=True,

        hide_index=True

    )


    st.subheader(
        "Model Performance Chart"
    )


    chart_data = results_df.melt(

        id_vars="Model",

        value_vars=metric_columns,

        var_name="Metric",

        value_name="Score"

    )


    fig = px.bar(

        chart_data,

        x="Model",

        y="Score",

        color="Metric",

        barmode="group",

        title="Model Performance Comparison"

    )


    fig.update_yaxes(
        range=[0, 1]
    )


    st.plotly_chart(

        fig,

        use_container_width=True

    )


    st.subheader(
        "Metric Explanation"
    )


    metric_explanation = pd.DataFrame({

        "Metric": [

            "Accuracy",

            "Precision",

            "Recall",

            "F1-Score",

            "ROC-AUC"

        ],

        "Meaning": [

            "Overall percentage of correct predictions",

            "How many predicted fraud cases were actually fraud",

            "How many actual fraud cases were detected",

            "Balance between precision and recall",

            "Overall classification performance"

        ]

    })


    st.dataframe(

        metric_explanation,

        use_container_width=True,

        hide_index=True

    )


# ============================================================
# END OF APPLICATION
# ============================================================