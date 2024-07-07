#!/bin/bash
source venv/bin/activate
python verify_cuda.py
streamlit run src/app.py