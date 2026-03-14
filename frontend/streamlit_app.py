"""
Finance AutoML Manager - Advanced Analytics Dashboard
A unified, single-flow interface for training, prediction, and visualization.
"""

from pathlib import Path
import sys
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

# Add repo root to path
_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from pipeline.preprocessing import clean_data
from pipeline.feature_engineering import create_features

# Page configuration
st.set_page_config(
    page_title="Finance AutoML Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        padding: 0.5rem 0;
        border-bottom: 2px solid #3498db;
        margin: 1.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📊 Finance AutoML Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Train, Predict & Visualize - All in One Place</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎮 Control Panel")
    
    with st.expander("Algorithm Settings", expanded=True):
        algorithm = st.selectbox(
            "🤖 Select ML Algorithm",
            [
                "Random Forest",
                "Logistic Regression",
                "Naive Bayes",
                "Decision Tree",
                "SVM",
            ],
            help="Choose the machine learning algorithm for training",
        )
        
        test_size = st.slider(
            "Test Split (%)",
            10,
            40,
            20,
            help="Percentage of data to use for testing",
        ) / 100
    
    with st.expander("Save Options", expanded=True):
        save_artifacts = st.checkbox(
            "💾 Save Model + Vectorizer",
            value=False,
            help="Save trained model to models/ directory",
        )
    
    st.markdown("---")
    st.markdown("### 📋 About")
    st.info("""
    **Supported Categories:**
    - 🍔 Food
    - 🚗 Transport  
    - 🛒 Shopping
    - 🥬 Groceries
    - 📄 Bills
    - 🎬 Subscription
    - ⛽ Fuel
    - 💸 Other
    """)

@st.cache_resource
def _load_artifacts():
    root_dir = Path(__file__).resolve().parents[1]
    model = joblib.load(root_dir / "models" / "best_model.pkl")
    vectorizer = joblib.load(root_dir / "models" / "vectorizer.pkl")
    return model, vectorizer


def render_visualization_tabs(df: pd.DataFrame, model=None, X_test=None, y_test=None, y_pred=None):
    """Render comprehensive visualization section with tabs."""
    
    viz_tabs = st.tabs([
        "📈 Basic",
        "🔥 Analytics", 
        "🎯 ML Insights",
        "🔍 PCA/t-SNE",
    ])
    
    # Tab 1: Basic Charts
    with viz_tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Category Distribution**")
            if "category" in df.columns:
                fig = px.pie(df, names="category", title="Spending by Category", hole=0.4)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**📊 Category Counts**")
            if "category" in df.columns:
                counts = df["category"].value_counts().reset_index()
                counts.columns = ["category", "count"]
                fig = px.bar(counts, x="category", y="count", color="category")
                st.plotly_chart(fig, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**📈 Histogram**")
            if "amount" in df.columns:
                fig = px.histogram(df, x="amount", title="Amount Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            st.markdown("**📦 Box Plot**")
            if "amount" in df.columns and "category" in df.columns:
                fig = px.box(df, x="category", y="amount", title="Amount by Category")
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Advanced Analytics
    with viz_tabs[1]:
        st.markdown("**🔥 Correlation Heatmap**")
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr = numeric_df.corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
            st.plotly_chart(fig, use_container_width=True)
        
        if "date" in df.columns:
            try:
                df["date"] = pd.to_datetime(df["date"])
                monthly = df.groupby(df["date"].dt.to_period("M")).agg({"amount": "sum"}).reset_index()
                monthly["date"] = monthly["date"].astype(str)
                fig = px.line(monthly, x="date", y="amount", title="Monthly Spending Trend", markers=True)
                st.plotly_chart(fig, use_container_width=True)
            except:
                pass
    
    # Tab 3: ML Insights
    with viz_tabs[2]:
        if model is not None and y_test is not None and y_pred is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎯 Confusion Matrix**")
                labels = sorted(pd.Series(y_test).astype(str).unique().tolist())
                cm = confusion_matrix(y_test.astype(str), pd.Series(y_pred).astype(str), labels=labels)
                cm_df = pd.DataFrame(cm, index=labels, columns=labels)
                fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**📈 ROC Curve**")
                try:
                    if hasattr(model, 'predict_proba'):
                        le = LabelEncoder()
                        y_test_encoded = le.fit_transform(y_test.astype(str))
                        y_pred_proba = model.predict_proba(X_test)
                        
                        fig = go.Figure()
                        for i, class_name in enumerate(le.classes_):
                            y_binary = (y_test_encoded == i).astype(int)
                            fpr, tpr, _ = roc_curve(y_binary, y_pred_proba[:, i])
                            roc_auc = auc(fpr, tpr)
                            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{class_name} (AUC={roc_auc:.2f})'))
                        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
                        fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
                        st.plotly_chart(fig, use_container_width=True)
                except:
                    st.info("ROC not available")
            
            # Feature Importance
            st.markdown("**🔍 Feature Importance**")
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                imp_df = pd.DataFrame({
                    'feature': [f"feature_{i}" for i in range(min(20, len(importances)))],
                    'importance': importances[:20],
                }).sort_values('importance', ascending=True)
                fig = px.bar(imp_df, x='importance', y='feature', orientation='h')
                st.plotly_chart(fig, use_container_width=True)
            elif hasattr(model, 'coef_'):
                coef = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
                coef_df = pd.DataFrame({
                    'feature': [f"feature_{i}" for i in range(min(20, len(coef)))],
                    'coefficient': coef[:20],
                }).sort_values('coefficient', ascending=True)
                fig = px.bar(coef_df, x='coefficient', y='feature', orientation='h')
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Dimensionality Reduction
    with viz_tabs[3]:
        if X_test is not None and y_test is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎯 PCA**")
                try:
                    X_dense = X_test.toarray() if hasattr(X_test, 'toarray') else X_test
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_dense)
                    pca_df = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'category': y_test})
                    fig = px.scatter(pca_df, x='PC1', y='PC2', color='category')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.info(f"PCA error: {e}")
            
            with col2:
                st.markdown("**🔍 t-SNE**")
                try:
                    X_dense = X_test.toarray() if hasattr(X_test, 'toarray') else X_test
                    if len(X_dense) > 200:
                        indices = np.random.choice(len(X_dense), 200, replace=False)
                        X_sample = X_dense[indices]
                        y_sample = y_test.iloc[indices] if hasattr(y_test, 'iloc') else y_test[indices]
                    else:
                        X_sample = X_dense
                        y_sample = y_test
                    
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_sample)-1))
                    X_tsne = tsne.fit_transform(X_sample)
                    tsne_df = pd.DataFrame({'t-SNE1': X_tsne[:, 0], 't-SNE2': X_tsne[:, 1], 'category': y_sample})
                    fig = px.scatter(tsne_df, x='t-SNE1', y='t-SNE2', color='category')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.info(f"t-SNE error: {e}")

# Initialize session state
if "result_df" not in st.session_state:
    st.session_state.result_df = None
if "model" not in st.session_state:
    st.session_state.model = None
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None
if "y_pred" not in st.session_state:
    st.session_state.y_pred = None

# Step 1: Upload
st.markdown('<h2 class="section-header">📤 Step 1: Upload Dataset</h2>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Drop your CSV file here", type=["csv"])

if not uploaded_file:
    st.info("👆 Please upload a CSV file to get started")
else:
    df = pd.read_csv(uploaded_file)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Total Rows", len(df))
    col2.metric("📋 Columns", len(df.columns))
    col3.metric("✅ Has Description", "Yes" if "description" in df.columns else "No")
    col4.metric("✅ Has Labels", "Yes" if "category" in df.columns else "No")
    
    with st.expander("👀 Preview Data", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
    
    if "description" not in df.columns:
        st.error("❌ CSV must contain a 'description' column")
    else:
        has_labels = "category" in df.columns
        
        # Step 2: Train/Predict
        st.markdown('<h2 class="section-header">🤖 Step 2: Process Data</h2>', unsafe_allow_html=True)
        
        if has_labels:
            st.info("📊 **Labeled dataset detected**. The app will train a model and evaluate performance.")
        else:
            st.info("🔮 **Unlabeled dataset detected**. The app will use a pre-trained model for predictions.")
        
        col_run, col_clear = st.columns([1, 5])
        with col_run:
            run_btn = st.button("🚀 Run", type="primary", use_container_width=True)
        with col_clear:
            if st.button("🔄 Clear", use_container_width=True):
                st.session_state.clear()
                st.rerun()
        
        if run_btn:
            with st.spinner("Processing..."):
                try:
                    if has_labels:
                        df_clean = clean_data(df)
                        X, y, vectorizer = create_features(df_clean)
                        
                        stratify_arg = y if y.nunique() > 1 else None
                        X_train, X_test_split, y_train, y_test_split = train_test_split(
                            X, y, test_size=test_size, random_state=42, stratify=stratify_arg
                        )
                        
                        if algorithm == "Logistic Regression":
                            model = LogisticRegression(max_iter=1000, random_state=42)
                        elif algorithm == "Naive Bayes":
                            model = MultinomialNB()
                        elif algorithm == "Decision Tree":
                            model = DecisionTreeClassifier(random_state=42)
                        elif algorithm == "SVM":
                            model = SVC(random_state=42, probability=True)
                        else:
                            model = RandomForestClassifier(n_estimators=200, random_state=42)
                        
                        model.fit(X_train, y_train)
                        y_pred_split = model.predict(X_test_split)
                        acc = accuracy_score(y_test_split, y_pred_split)
                        
                        st.session_state.model = model
                        st.session_state.X_test = X_test_split
                        st.session_state.y_test = y_test_split
                        st.session_state.y_pred = y_pred_split
                        st.session_state.result_df = df_clean.copy()
                        
                        if save_artifacts:
                            model_dir = _ROOT_DIR / "models"
                            model_dir.mkdir(parents=True, exist_ok=True)
                            joblib.dump(model, model_dir / "best_model.pkl")
                            joblib.dump(vectorizer, model_dir / "vectorizer.pkl")
                        
                        # Results
                        st.markdown('<h2 class="section-header">📊 Step 3: Training Results</h2>', unsafe_allow_html=True)
                        
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("🎯 Algorithm", algorithm)
                        m2.metric("📈 Accuracy", f"{acc:.2%}")
                        m3.metric("📊 Classes", y.nunique())
                        m4.metric("🧪 Test Size", len(y_test_split))
                        
                        with st.expander("📋 Classification Report"):
                            st.text(classification_report(y_test_split, y_pred_split, zero_division=0))
                        
                        if save_artifacts:
                            st.success("✅ Model saved to models/ directory")
                    
                    else:
                        model, vectorizer = _load_artifacts()
                        X = vectorizer.transform(df["description"].astype(str))
                        preds = model.predict(X)
                        
                        result_df = df.copy()
                        result_df["predicted_category"] = preds
                        st.session_state.result_df = result_df
                        st.session_state.model = None
                        st.session_state.X_test = None
                        st.session_state.y_test = None
                        st.session_state.y_pred = None
                        
                        st.success(f"✅ Predictions complete for {len(result_df)} transactions")
                
                except FileNotFoundError:
                    st.error("❌ Pre-trained model not found. Train with labeled data first.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Step 3 & 4: Output & Visualizations
        if st.session_state.result_df is not None:
            result_df = st.session_state.result_df
            
            st.markdown('<h2 class="section-header">📋 Step 3: Output Data</h2>', unsafe_allow_html=True)
            st.dataframe(result_df, use_container_width=True)
            
            csv = result_df.to_csv(index=False)
            st.download_button("📥 Download CSV", data=csv, 
                           file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv")
            
            st.markdown('<h2 class="section-header">📈 Step 4: Visualizations</h2>', unsafe_allow_html=True)
            render_visualization_tabs(
                result_df,
                model=st.session_state.get("model"),
                X_test=st.session_state.get("X_test"),
                y_test=st.session_state.get("y_test"),
                y_pred=st.session_state.get("y_pred"),
            )

st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>Built with ❤️ using Streamlit, Plotly & Scikit-learn</p>", unsafe_allow_html=True)
