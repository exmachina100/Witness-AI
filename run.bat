@echo off
call venv\Scripts\activate.bat
python verify_cuda.py
streamlit run src\app.py
pause