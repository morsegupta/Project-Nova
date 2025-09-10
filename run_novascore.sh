#!/bin/bash

if [ ! -d "novascore_env" ]; then
    python -m venv novascore_env
    source novascore_env/bin/activate
    brew install cmake
    brew install ninja
    brew install libomp
    pip install -r requirements.txt
else
    source novascore_env/bin/activate
fi

osascript <<EOF
tell application "Terminal"
    do script "cd $(pwd); source novascore_env/bin/activate; python nova_api.py"
end tell
EOF

osascript <<EOF
tell application "Terminal"
    do script "cd $(pwd); source novascore_env/bin/activate; echo '' | streamlit run app.py"
end tell
EOF
