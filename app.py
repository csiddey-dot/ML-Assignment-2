import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Machine Learning Classification Evaluator")
st.write("Evaluate different classification models on the Breast Cancer diagnostic dataset.")

# 1. Dataset Upload Option
uploaded_file = st.file_uploader("Upload your test_data.csv", type=["csv"])

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.success("Test Data Loaded Successfully!")
    
    # Separate features and target
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']
    
    # Load the scaler
    scaler = joblib.load('model/scaler.pkl')
    X_test_scaled = scaler.transform(X_test)
    
    # 2. Model Selection Dropdown
    # These names exactly match the .pkl files we generated
    model_name = st.selectbox("Select a Model to Evaluate", 
                              ["Logistic_Regression", "Decision_Tree", "KNN", "Naive_Bayes", "Support_Vector_Machine"])
    
    if st.button("Evaluate Model"):
        # Load the selected model
        model = joblib.load(f"model/{model_name}.pkl")
        
        # Generate predictions
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # 3. Display Evaluation Metrics
        st.subheader(f"Evaluation Metrics: {model_name.replace('_', ' ')}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", round(accuracy_score(y_test, y_pred), 4))
        col2.metric("AUC", round(roc_auc_score(y_test, y_proba), 4))
        col3.metric("Precision", round(precision_score(y_test, y_pred), 4))
        
        col4, col5, col6 = st.columns(3)
        col4.metric("Recall", round(recall_score(y_test, y_pred), 4))
        col5.metric("F1 Score", round(f1_score(y_test, y_pred), 4))
        col6.metric("MCC", round(matthews_corrcoef(y_test, y_pred), 4))
        
        # 4. Display Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        st.pyplot(fig)