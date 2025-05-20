
import os
import sys
import streamlit as st

# Disable Streamlit's file watcher
os.environ["STREAMLIT_SERVER_WATCH_PATTERNS"] = ""

st.title("Debug App")
st.write("Initial loading complete")

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Log the directory structure
st.write(f"Current directory: {current_dir}")
st.write(f"Parent directory: {parent_dir}")
st.write(f"Files in current dir: {os.listdir(current_dir)}")
st.write(f"Files in parent dir: {os.listdir(parent_dir)}")

# Try to import torch
try:
    import torch
    st.write(f"PyTorch version: {torch.__version__}")
    st.write(f"CUDA available: {torch.cuda.is_available()}")
    if hasattr(torch.backends, 'mps'):
        st.write(f"MPS available: {torch.backends.mps.is_available()}")
except Exception as e:
    st.error(f"Error importing torch: {e}")
    st.exception(e)

# Try to check the output directory
try:
    output_dir = os.path.join(parent_dir, "output")
    if os.path.exists(output_dir):
        st.write(f"Output directory exists: {output_dir}")
        st.write(f"Files in output: {os.listdir(output_dir)}")
    else:
        st.error(f"Output directory doesn't exist: {output_dir}")
except Exception as e:
    st.error(f"Error checking output directory: {e}")
    st.exception(e)

# Try to load configs
try:
    st.write("Importing configs...")
    from configs.base_config import Config
    st.write("Config imported successfully")
except Exception as e:
    st.error(f"Error importing configs: {e}")
    st.exception(e)

# Try to load model factory
try:
    st.write("Importing model_factory...")
    from models.model_factory import get_best_device
    device = get_best_device()
    st.write(f"Best device: {device}")
except Exception as e:
    st.error(f"Error importing model_factory: {e}")
    st.exception(e)

st.write("Debug complete")


import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional



