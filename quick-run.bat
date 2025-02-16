REM Launch the application
streamlit run app.py
if errorlevel 1 (
    echo Error: Failed to start Streamlit app
    pause
)
