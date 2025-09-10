if (-Not (Test-Path ".\novascore_env")) {
    python -m venv novascore_env
    ./novascore_env/Scripts/activate
    pip install cmake
    pip install ninja
    pip install libomp
    pip install -r requirements.txt
} else {
    ./novascore_env/Scripts/activate
}

Start-Process powershell -ArgumentList "-NoExit", "-Command", "./novascore_env/Scripts/activate; python nova_api.py"

Start-Process powershell -ArgumentList "-NoExit", "-Command", "./novascore_env/Scripts/activate; echo '' | streamlit run app.py"
