import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import tensorflow as tf
from scipy.ndimage import measurements
from tensorflow.keras.models import load_model
from streamlit_extras.metric_cards import style_metric_cards

st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="✍️",
    layout="wide"
)

st.markdown("""
<style>
    .metric-card {
        background: #f0f2f6;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: scale(1.02);
    }
    .metric-title {
        color: #34495e;
        font-size: 16px;
        font-weight: 600;
    }
    .metric-value {
        color: #2c3e50;
        font-size: 30px;
        font-weight: 700;
    }
    .header {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    .model-card {
        background: #ffffff;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        margin: 20px 0;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

st.title("✍️ Handwritten Digit Recognition System")
st.markdown("""
        <div align = "right">
        <p style="color:#7f8c8d;font-size:14px;">
            By,<br>
            <b>T004 Rajdeep Thorat</b><br>
            <b>T017 Rohan Patil</b><br>
        </p>
        </div>
        """, unsafe_allow_html=True)

@st.cache_resource

def load_models_and_metrics():
    svm_model = joblib.load('mnist_svm_model.joblib')
    cnn_model = load_model('mnist_cnn_model.h5')

    svm_confusion = np.array([
        [975, 0, 2, 0, 1, 0, 2, 1, 2, 0],
        [0, 1144, 3, 1, 1, 0, 0, 3, 0, 0],
        [3, 2, 950, 1, 1, 1, 2, 3, 4, 0],
        [0, 1, 8, 1007, 2, 7, 0, 3, 4, 2],
        [3, 0, 0, 0, 890, 0, 0, 2, 1, 10],
        [0, 0, 0, 8, 1, 920, 5, 0, 3, 0],
        [1, 0, 0, 0, 4, 2, 953, 0, 1, 0],
        [0, 3, 9, 0, 3, 0, 0, 1038, 0, 2],
        [1, 2, 5, 7, 1, 3, 4, 5, 935, 6],
        [6, 2, 0, 6, 8, 1, 0, 6, 5, 1002]
    ])

    cnn_confusion = np.array([
        [979, 0, 0, 0, 0, 0, 1, 2, 1, 0],
        [0, 1148, 1, 0, 0, 0, 0, 2, 0, 1],
        [0, 0, 964, 1, 0, 0, 1, 1, 0, 0],
        [2, 0, 6, 1017, 0, 4, 0, 1, 1, 3],
        [0, 0, 0, 0, 899, 0, 0, 0, 0, 7],
        [0, 0, 0, 5, 0, 926, 5, 0, 1, 0],
        [0, 0, 0, 0, 2, 1, 957, 0, 1, 0],
        [0, 3, 5, 0, 1, 0, 0, 1042, 0, 4],
        [2, 0, 5, 2, 1, 4, 1, 0, 948, 6],
        [4, 2, 0, 0, 4, 2, 0, 2, 1, 1021]
    ])

    svm_metrics = {
        'accuracy': 0.9814,
        'precision': 0.9812,
        'recall': 0.9813,
        'f1_score': 0.9813,
        'confusion_matrix': svm_confusion
    }

    cnn_metrics = {
        'accuracy': 0.9901,
        'precision': 0.9901,
        'recall': 0.9901,
        'f1_score': 0.9901,
        'confusion_matrix': cnn_confusion
    }

    return svm_model, cnn_model, svm_metrics, cnn_metrics

svm_model, cnn_model, svm_metrics, cnn_metrics = load_models_and_metrics()

model_type = st.sidebar.radio("🔍 Choose Model:", ["Support Vector Machine", "Convolutional Neural Network"])

if model_type == "Support Vector Machine":
    model = svm_model
    metrics = svm_metrics
else:
    model = cnn_model
    metrics = cnn_metrics

def center_digit(digit_array):
    # Create a copy to avoid modifying original
    digit = digit_array.copy()
    
    # Threshold to get binary image (handle inverted MNIST format)
    threshold = np.percentile(digit, 95)  # Use top 5% as threshold
    binary = digit > threshold
    
    # Calculate center of mass (only using pixels above threshold)
    try:
        cy, cx = measurements.center_of_mass(binary)
    except:
        return digit  # Return original if center calculation fails
    
    # Handle case where all pixels are background
    if np.isnan(cx) or np.isnan(cy):
        return digit
    
    # Calculate needed shift
    rows, cols = digit.shape
    shiftx = int(np.round(cols/2.0 - cx))
    shifty = int(np.round(rows/2.0 - cy))
    
    shifted = np.zeros_like(digit)
    x_start = max(shiftx, 0)
    x_end = min(cols + shiftx, cols)
    y_start = max(shifty, 0)
    y_end = min(rows + shifty, rows)
    
    orig_x_start = max(-shiftx, 0)
    orig_x_end = min(cols - shiftx, cols)
    orig_y_start = max(-shifty, 0)
    orig_y_end = min(rows - shifty, rows)
    
    shifted[y_start:y_end, x_start:x_end] = digit[orig_y_start:orig_y_end, orig_x_start:orig_x_end]
    
    return shifted

def preprocess_image(image_data, model_type, center_digits=True):
    img = Image.fromarray(image_data.astype('uint8')).convert('L')
    img = img.resize((28, 28), Image.LANCZOS)
    img_array = 255 - np.array(img)  # BIG INVERT
    
    if center_digits:
        img_array = center_digit(img_array)
    
    if model_type == "Convolutional Neural Network":
        img_array = img_array.reshape(1, 28, 28, 1) / 255.0
    else:  
        img_array = img_array.flatten().reshape(1, -1) / 255.0
    
    return img_array

col1, col2 = st.columns([2, 3])

with col1:
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

    if st.button("Recognize Digit", type="primary", use_container_width=True):
        if canvas.image_data is not None:
            with st.spinner('Analyzing...'):
                processed_img = preprocess_image(canvas.image_data, model_type)
                if model_type == "Convolutional Neural Network":
                    prediction = np.argmax(model.predict(processed_img), axis=1)[0]
                    probabilities = model.predict(processed_img)[0]
                else:
                    prediction = model.predict(processed_img)[0]
                    probabilities = model.predict_proba(processed_img)[0]

                st.success(f"Predicted Digit: **{prediction}**")

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
            st.warning("⚠️ Please draw a digit first")

with col2:
    st.subheader("📈 Model Performance Dashboard", divider='blue')

    cols = st.columns(4)
    for i, metric in enumerate(['accuracy', 'precision', 'recall', 'f1_score']):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{metric.replace('_', ' ').title()}</div>
                <div class="metric-value">{metrics[metric]:.2%}</div>
            </div>
            """, unsafe_allow_html=True)

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

    # For SVM model
    svm_content = """
    <p>
    <ul>
        <li><b>Kernel:</b> Radial Basis Function (RBF)<br>
        <li><b>Regularization (C):</b> 5.0
    </ul>
    <p>
    <p>Trained on the MNIST dataset containing 60,000 handwritten digit samples.</p>
    """

    # For CNN model
    cnn_content = """
    <p>
    <ul>
        <li><b>Layers:</b> Conv2D -> MaxPooling -> Conv2D -> MaxPooling -> Dense <br>
        <li><b>Optimizer:</b> Adam
    <ul>
    </p>
    <p>Trained on the MNIST dataset with augmented handwritten digits.</p>
    """

    content = svm_content if model_type == "Support Vector Machine" else cnn_content

    st.markdown(f"""
        <div class="model-card">
            <h3>🧠 Model Architecture</h3>
            <p>This digit recognition system uses a <b>{model_type}</b> model.</p>
            {content}
        """, unsafe_allow_html=True)

# --- Comparison Section ---
st.markdown("---")
st.subheader("🔍 Model Performance Comparison")

# Create two columns for side-by-side metrics
col1, col2 = st.columns(2)

# SVM Metrics Card
with col1:
    st.markdown("### 🖥️ SVM Model")
    st.metric("Accuracy", f"{svm_metrics['accuracy']:.2%}")
    st.metric("Precision", f"{svm_metrics['precision']:.2%}")
    st.metric("Recall", f"{svm_metrics['recall']:.2%}")
    st.metric("F1 Score", f"{svm_metrics['f1_score']:.2%}")

# CNN Metrics Card
with col2:
    st.markdown("### 🧠 CNN Model")
    st.metric("Accuracy", f"{cnn_metrics['accuracy']:.2%}")
    st.metric("Precision", f"{cnn_metrics['precision']:.2%}")
    st.metric("Recall", f"{cnn_metrics['recall']:.2%}")
    st.metric("F1 Score", f"{cnn_metrics['f1_score']:.2%}")

# Style the metric cards
style_metric_cards()

# --- Performance Visualization ---
st.markdown("### 📊 Performance Metrics Comparison")

# Prepare data for visualization
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'] * 2,
    'Value': [
        svm_metrics['accuracy'], svm_metrics['precision'], svm_metrics['recall'], svm_metrics['f1_score'],
        cnn_metrics['accuracy'], cnn_metrics['precision'], cnn_metrics['recall'], cnn_metrics['f1_score']
    ],
    'Model': ['SVM'] * 4 + ['CNN'] * 4
})

# Create interactive bar chart
fig = px.bar(metrics_df, 
             x='Metric', 
             y='Value', 
             color='Model',
             barmode='group',
             text_auto='.2%',
             title='Model Performance Metrics Comparison',
             color_discrete_map={'SVM': '#636EFA', 'CNN': '#EF553B'})

fig.update_layout(yaxis_tickformat=".0%",
                  yaxis_range=[0, 1.05],
                  hovermode="x unified")

st.plotly_chart(fig, use_container_width=True)

# --- Detailed Comparison Table ---
st.markdown("### 📝 Detailed Metrics Table")

comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'SVM': [svm_metrics['accuracy'], svm_metrics['precision'], svm_metrics['recall'], svm_metrics['f1_score']],
    'CNN': [cnn_metrics['accuracy'], cnn_metrics['precision'], cnn_metrics['recall'], cnn_metrics['f1_score']]
})

# Add difference column
comparison_df['Difference'] = comparison_df['CNN'] - comparison_df['SVM']

# Format the table
styled_df = (comparison_df.set_index('Metric')
             .style
             .format("{:.2%}", subset=['SVM', 'CNN'])
             .format("{:+.2%}", subset=['Difference'])
             .background_gradient(cmap='RdYlGn', subset=['Difference'], vmin=-0.2, vmax=0.2)
             .set_properties(**{'text-align': 'center'}))

st.dataframe(styled_df)

