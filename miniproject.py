import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="✍️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-title {
        color: #2c3e50;
        font-size: 16px;
        font-weight: 600;
    }
    .metric-value {
        color: #000000;
        font-size: 28px;
        font-weight: 700;
    }
    .header {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    .model-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

st.title("✍️ Handwritten Digit Recognition System")
st.markdown("""
        <div align = "right">
        <p>
            By,<br>
            T004 Rajdeep Thorat<br>
            T017 Rohan Patil<br>
        </p>
        </div>
        """, unsafe_allow_html=True)

@st.cache_resource
def load_model_and_metrics():
    model = joblib.load('mnist_svm_model.joblib')
    
    metrics = {
        'accuracy': 0.9965,
        'precision': 0.9965,
        'recall': 0.9965,
        'f1_score': 0.9965,
        'confusion_matrix': np.array([
 [979, 0, 0, 0, 0, 0, 0, 1, 0, 0],
 [0, 1133, 1, 0, 0, 0, 0, 1, 0, 0],
 [1, 0, 1029, 0, 0, 0, 0, 0, 2, 0],
 [0, 0, 1, 1008, 0, 1, 0, 0, 0, 0],
 [1, 0, 0, 0, 980, 0, 0, 0, 0, 1],
 [0, 0, 0, 2, 1, 888, 0, 0, 1, 0],
 [1, 1, 0, 0, 0, 1, 954, 0, 1, 0],
 [0, 1, 2, 0, 0, 0, 0, 1025, 0, 0],
 [0, 0, 0, 2, 1, 1, 0, 1, 967, 2],
 [1, 1, 0, 2, 2, 0, 0, 0, 1, 1002]])
    }
    return model, metrics

model, metrics = load_model_and_metrics()

# Main app columns
col1, col2 = st.columns([2, 3])

with col1:
    # Drawing canvas
    st.subheader("Draw a Digit")
    canvas = st_canvas(
        fill_color="white",
        stroke_width=18,
        stroke_color="black",
        background_color="white",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
        display_toolbar=True
    )
    def preprocess_image(image_data):
        img = Image.fromarray(image_data.astype('uint8')).convert('L')
        img = img.resize((28, 28), Image.LANCZOS)
        img_array = 255 - np.array(img)  # Invert to white-on-black
        return img_array.flatten().reshape(1, -1) / 255.0
    # Prediction buttons
    if st.button("Recognize Digit", type="primary", use_container_width=True):
        if canvas.image_data is not None:
            with st.spinner('Analyzing...'):
                processed_img = preprocess_image(canvas.image_data)
                prediction = model.predict(processed_img)[0]
                probabilities = model.predict_proba(processed_img)[0]
                
                # Results
                st.success(f"Predicted Digit: **{prediction}**")
                
                # Confidence gauge
                fig = px.bar(
                    x=[f"Digit {i}" for i in range(10)],
                    y=probabilities*100,
                    labels={'x': 'Digit', 'y': 'Confidence (%)'},
                    text=[f"{p:.1f}%" for p in probabilities*100],
                    color=probabilities,
                    color_continuous_scale='Blues'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please draw a digit first")

with col2:
    # Model Performance Dashboard
    st.subheader("Model Performance Dashboard", divider='blue')
    
    # Metrics cards
    cols = st.columns(4)
    with cols[0]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Accuracy</div>
            <div class="metric-value">{:.2%}</div>
        </div>
        """.format(metrics['accuracy']), unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Precision</div>
            <div class="metric-value">{:.2%}</div>
        </div>
        """.format(metrics['precision']), unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Recall</div>
            <div class="metric-value">{:.2%}</div>
        </div>
        """.format(metrics['recall']), unsafe_allow_html=True)
    
    with cols[3]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">F1 Score</div>
            <div class="metric-value">{:.2%}</div>
        </div>
        """.format(metrics['f1_score']), unsafe_allow_html=True)
    
    # Confusion matrix heatmap
    st.markdown("#### Confusion Matrix")
    fig = px.imshow(
        metrics['confusion_matrix'],
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=[str(i) for i in range(10)],
        y=[str(i) for i in range(10)],
        text_auto=True,
        aspect="auto",
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Model description
    st.markdown("""
    <div class="model-card">
        <h3>Model Architecture</h3>
        <p>This digit recognition system uses a <b>Support Vector Machine (SVM)</b> with:</p>
        <ul>
            <li><b>Kernel:</b> Radial Basis Function (RBF)</li>
            <li><b>Regularization (C):</b> 5.0</li>
        </ul>
        <p>Trained on the MNIST dataset containing 60,000 handwritten digit samples.</p>
    </div>
    """, unsafe_allow_html=True)

# Preprocessing function
def preprocess_image(image_data):
    img = Image.fromarray(image_data.astype('uint8')).convert('L')
    img = img.resize((28, 28), Image.LANCZOS)
    img_array = 255 - np.array(img)  # Invert to white-on-black
    return img_array.flatten().reshape(1, -1) / 255.0