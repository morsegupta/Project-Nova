#!/bin/bash

if [ ! -d "novascore_env" ]; then
    python -m venv novascore_env
    source novascore_env/bin/activate
    pip install cmake
    pip install ninja
    pip install libomp
    pip install -r requirements.txt
else
    source novascore_env/bin/activate
fi

TERMINAL=${TERMINAL:-gnome-terminal}

$TERMINAL -- bash -c "cd $(pwd); source novascore_env/bin/activate; python nova_api.py; exec bash" &

$TERMINAL -- bash -c "cd $(pwd); source novascore_env/bin/activate; echo '' | streamlit run app.py; exec bash" &
