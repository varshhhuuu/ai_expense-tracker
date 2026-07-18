# app.py - AI-Powered Expense Tracker with Chatbot
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from transformers import pipeline
# ==============================
# CONFIG
# ==============================
DATA_FILE = "expenses.csv"
MODEL_FILE = "category_model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"
# ==============================
# 1. LOAD OR CREATE DATASET
# ==============================
def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        # Ensure date is string
        if 'date' in df.columns:
            df['date'] = df['date'].astype(str)
        return df
    else:
        return pd.DataFrame(columns=["date", "description", "amount", "category"])
def save_data(df):
    df.to_csv(DATA_FILE, index=False)
# Load data
if 'df' not in st.session_state:
    st.session_state.df = load_data()
df = st.session_state.df
# ==============================
# 2. TRAIN ML MODEL FOR CATEGORY PREDICTION
# ==============================
# ==============================
# 2. TRAIN ML MODEL – Random Forest vs Multinomial Naive Bayes
# ==============================
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

def train_models():
    if len(df) < 10:
        return None, None, None, "None"
    
    X = df['description'].fillna("").str.lower()
    y = df['category']
    
    # Encode labels (needed only for comparison, not for Naive Bayes directly)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Better TF-IDF settings for short text
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=500,
        ngram_range=(1, 3),          # unigrams + bigrams + trigrams → huge accuracy boost
        min_df=1
    )
    X_vec = vectorizer.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
    )
    
    # Model 1 – Random Forest (your original)
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    
    # Model 2 – Multinomial Naive Bayes (new!)
    nb = MultinomialNB(alpha=0.5)          # alpha = smoothing (0.1–1.0 works great)
    nb.fit(X_train, y_train)
    nb_acc = nb.score(X_test, y_test)
    
    # Pick the winner
    if nb_acc >= rf_acc:
        best_model = nb
        winner = "Naive Bayes"
        st.success(f"AI Model trained → Winner: **{winner}** (Accuracy: {nb_acc:.1%})")
    else:
        best_model = rf
        winner = "Random Forest"
        st.success(f"AI Model trained → Winner: **{winner}** (Accuracy: {rf_acc:.1%})")
    
    # Save everything
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(best_model, f)
    with open(VECTORIZER_FILE, 'wb') as f:
        pickle.dump(vectorizer, f)
    with open("label_encoder.pkl", 'wb') as f:
        pickle.dump(le, f)
    
    return best_model, vectorizer, le, winner

# Load or train
model, vectorizer, label_encoder, active_model = None, None, None, "None"
if len(df) >= 10:
    try:
        model, vectorizer, label_encoder, active_model = train_models()
        st.sidebar.success(f"Active AI Model: **{active_model}**")
    except Exception as e:
        st.warning(f"Training failed: {e}")
else:
    st.info(f"Add {10 - len(df)} more expenses to unlock AI prediction.")

# Updated prediction function
def predict_category(description):
    files = [MODEL_FILE, VECTORIZER_FILE, "label_encoder.pkl"]
    if not all(os.path.exists(f) for f in files):
        return "Other"
    try:
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_FILE, 'rb') as f:
            vectorizer = pickle.load(f)
        with open("label_encoder.pkl", 'rb') as f:
            le = pickle.load(f)
        
        vec = vectorizer.transform([description.lower()])
        pred_num = model.predict(vec)[0]
        return le.inverse_transform([pred_num])[0]
    except:
        return "Other"
# ==============================
# 3. STREAMLIT UI
# ==============================
st.set_page_config(page_title="AI Expense Tracker", layout="wide")
st.title("AI-Powered Expense Tracker")
st.markdown("Track expenses, get insights, and chat with AI!")
tab1, tab2, tab3, tab4 = st.tabs(["Add Expense", "View & Analyze", "AI Chatbot", "Export"])
# -----------------------------
# TAB 1: Add Expense
# -----------------------------
with tab1:
    st.header("Add New Expense")
    with st.form("expense_form"):
        date = st.date_input("Date", value=datetime.today())
        description = st.text_input("Description", placeholder="e.g., Lunch at Cafe")
        amount = st.number_input("Amount ($)", min_value=0.01, format="%.2f")
        category_options = ["Food", "Transport", "Entertainment", "Shopping", "Bills", "Other"]
        # AI Suggestion
        suggested_category = "Other"
        if description.strip() and len(df) >= 10:
            suggested_category = predict_category(description)
            st.info(f"AI Suggests: **{suggested_category}**")
        category = st.selectbox("Category", category_options, index=category_options.index(suggested_category) if suggested_category in category_options else 0)
        submitted = st.form_submit_button("Add Expense")
        if submitted:
            new_row = pd.DataFrame([{
                "date": date.strftime('%Y-%m-%d'),
                "description": description,
                "amount": amount,
                "category": category
            }])
            st.session_state.df = pd.concat([df, new_row], ignore_index=True)
            save_data(st.session_state.df)
            st.success("Expense added successfully!")
            st.rerun()
# -----------------------------
# TAB 2: View & Analyze
# -----------------------------
with tab2:
    st.header("Your Expenses")
    if df.empty:
        st.info("No expenses yet. Add one!")
    else:
        # Prepare visualization copy
        df_viz = df.copy()
        df_viz["date"] = pd.to_datetime(df_viz["date"], errors="coerce")
        df_viz = df_viz.dropna(subset=["date"])
        # Recent Transactions
        st.subheader("Recent Transactions")
        recent = df_viz.sort_values("date", ascending=False).head(20).copy()
        recent["date"] = recent["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(recent, use_container_width=True)
        # Summary Metrics
        total = df_viz["amount"].sum()
        st.metric("Total Spent", f"${total:,.2f}")
        col1, col2 = st.columns(2)
        with col1:
            this_month = df_viz[df_viz["date"].dt.month == datetime.now().month]["amount"].sum()
            st.metric("This Month", f"${this_month:,.2f}")
        with col2:
            st.metric("Transactions", len(df_viz))
        # Spending by Category
        st.subheader("Spending by Category")
        cat_sum = df_viz.groupby("category")["amount"].sum().sort_values(ascending=False)
        fig_cat, ax_cat = plt.subplots(figsize=(8, max(4, len(cat_sum) * 0.4)))
        sns.barplot(x=cat_sum.values, y=cat_sum.index, palette="viridis", ax=ax_cat)
        ax_cat.set_xlabel("Amount ($)")
        ax_cat.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig_cat)
        # Monthly Trend
        st.subheader("Monthly Trend")
        df_viz["month"] = df_viz["date"].dt.to_period("M")
        monthly = df_viz.groupby("month")["amount"].sum()
        if not monthly.empty:
            fig_trend, ax_trend = plt.subplots(figsize=(9, 4))
            monthly.plot(kind="line", marker="o", ax=ax_trend, color="#1f77b4")
            ax_trend.set_ylabel("Total Spent ($)")
            ax_trend.set_title("Monthly Spending")
            ax_trend.grid(True, linestyle="--", alpha=0.5)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig_trend)
        else:
            st.info("Not enough data for monthly trend.")
# -----------------------------
# TAB 3: AI Chatbot
# -----------------------------
with tab3:
    st.header("Ask Your Expenses")
    @st.cache_resource
    def load_chatbot():
        try:
            return pipeline("text-generation", model="microsoft/DialoGPT-small")
        except:
            return None
    chatbot = load_chatbot()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    user_input = st.text_input("Ask: 'How much on food?' or 'Total spent?'")
    if st.button("Send") and user_input:
        lower = user_input.lower()
        response = ""
        # Rule-based responses
        if any(k in lower for k in ["food", "eat", "lunch", "dinner"]):
            spent = df[df['category'] == 'Food']['amount'].sum()
            response = f"Food: **${spent:,.2f}**"
        elif "total" in lower:
            response = f"Total: **${df['amount'].sum():,.2f}**"
        elif "month" in lower:
            this_month = df[pd.to_datetime(df['date']).dt.month == datetime.now().month]['amount'].sum()
            response = f"This month: **${this_month:,.2f}**"
        elif "transport" in lower:
            spent = df[df['category'] == 'Transport']['amount'].sum()
            response = f"Transport: **${spent:,.2f}**"
        else:
            if chatbot:
                try:
                    output = chatbot(user_input, max_length=100, truncation=True)
                    response = output[0]['generated_text'].split(user_input)[-1].strip()
                    if len(response) < 5:
                        response = "Try asking about food, total, or month."
                except:
                    response = "AI is thinking..."
            else:
                response = "Chatbot not available."
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))
    # Display chat
    for sender, msg in st.session_state.chat_history[-10:]:
        if sender == "You":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Bot:** {msg}")
# -----------------------------
# TAB 4: Export
# -----------------------------
with tab4:
    st.header("Export Data")
    csv = df.to_csv(index=False).encode()
    st.download_button(
        label="Download Expenses as CSV",
        data=csv,
        file_name=f"expenses_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("Tips")
st.sidebar.info("""
- Add **10+ expenses** to unlock AI prediction
- Use **chatbot** for quick insights
- Data saved in `expenses.csv`
- Export anytime!
""")