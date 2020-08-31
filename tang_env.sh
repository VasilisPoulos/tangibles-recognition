#!/bin/bash
echo 'Give a name for the tangibles python environment'
read env_name
python -m venv $env_name
source $env_name/bin/activate

pip install --upgrade pip
pip install numpy
pip install matplotlib
pip install opencv-python==4.1.2.30
pip install pytesseract
pip install imutils
pip install anytree
sudo apt-get install tesseract-ocr
