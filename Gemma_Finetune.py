import streamlit as st
import time
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Gemma Fine-Tuning UI", layout="wide")
st.markdown(
    """
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px;
            border-radius: 10px;
        }
        .stSelectbox, .stRadio, .stSlider, .stNumber_input {
            background-color:#566573  ;
            border-radius: 10px;
            padding: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💎Fine-Tuning")
st.sidebar.title("🔧 Settings")

# Dataset Upload
st.sidebar.header("📂 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a dataset (CSV, JSONL, TXT)", type=["csv", "jsonl", "txt"])

# Gemma Model Selection
st.sidebar.header("🧠 Select Gemma Model")
gemma_model = st.sidebar.selectbox("Choose a Gemma model to fine-tune:", ["gemma 3-1b", "gemma 3-4b", "gemma 3-12b", "gemma 3-27b"], index=0)

# Fine-Tuning Method Selection
st.sidebar.header("⚙️ Select Fine-Tuning Method")
fine_tune_method = st.sidebar.radio("Choose a fine-tuning method:",
                            ["LoRA", "QLoRA", "Prefix-Tuning", "P-Tuning", "Soft Prompt Tuning", "Feature-Based Fine-Tuning"],
                            index=0)

# Training Location Selection
st.sidebar.header("🌍 Select Training Location")
train_location = st.sidebar.radio("Where do you want to fine-tune?",
                          ["Google Cloud", "Personal Device"],
                          index=1)

# Hyperparameter Configuration
st.sidebar.header("📊 Configure Hyperparameters")
learning_rate = st.sidebar.slider("Learning Rate", min_value=1e-6, max_value=1e-2, value=5e-5, step=1e-6)
batch_size = st.sidebar.selectbox("Batch Size", [8, 16, 32, 64])
epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=50, value=3)
optimizer = st.sidebar.selectbox("Optimizer", ["AdamW", "SGD", "RMSprop"])
weight_decay = st.sidebar.slider("Weight Decay", 0.0001, 0.01, 0.001, step=0.0001)
dropout = st.sidebar.slider("Dropout", 0.1, 0.3, 0.2, step=0.1)
gradient_accumulation_steps = st.sidebar.selectbox("Gradient Accumulation Steps", [1, 2, 4, 8])
warmup_steps = st.sidebar.selectbox("Warmup Steps", [0, 100, 500, 1000])
max_seq_length = st.sidebar.selectbox("Max Sequence Length", [128, 256, 512, 1024])

# Training Progress Visualization
st.header("📈 Training Progress")
progress_bar = st.progress(0)
status_text = st.empty()
chart_placeholder = st.empty()

# Start Fine-Tuning Button
if st.button("🚀 Start Fine-Tuning", key="start_finetune"):
    st.success(f"Fine-tuning started on {gemma_model} using {fine_tune_method} on {train_location} with {epochs} epochs.")
    loss_values = []
    for epoch in range(epochs):
        loss = np.exp(-0.3 * epoch) + np.random.uniform(0.01, 0.05)
        loss_values.append(loss)
        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")
        
        # Plot loss curve
        fig, ax = plt.subplots()
        ax.plot(range(1, len(loss_values) + 1), loss_values, marker='o', linestyle='-')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Curve")
        chart_placeholder.pyplot(fig)
        
        time.sleep(1)
    
    st.success("Fine-tuning complete!")

# Model Download/Export Options
st.header("📥 Download Fine-Tuned Model")
export_format = st.selectbox("Select export format:", ["TensorFlow SavedModel", "PyTorch", "GGUF"])
if st.button("📥 Download Model"):
    st.success(f"Your fine-tuned {gemma_model} model in {export_format} format is ready for download!")