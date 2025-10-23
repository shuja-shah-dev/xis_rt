# Setup Instructions

## Create and Activate Virtual Environment
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

## Install all dependencies
pip install -r Install/req.txt

# Install TensorRT and cuDNN
Install these for respective python version. Recommended 3.10

# Run software
```bash
cd src/app/frontend
npm i
npm run build
cd ../../
python app.py
```

