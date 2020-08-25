# Tangibles GSoC repo
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/VasilisPoulos/tangibles-recognition/master)

## How to run 

Use the `tang_env.sh` script as:

```
source tang_env.sh
```

This will create a python environment in the same directory as the python script 
with all the required libraries installed.

Or, do the same process manually:

```
python -m venv tang-env
source tang-env

pip install --upgrade pip
pip install numpy
pip install matplotlib
pip install opencv-python==3.4.11.41
pip install pytesseract
pip install imutils
pip install anytree
sudo apt-get install tesseract-ocr
```

To run the tangibles python script on a test image use:

```
python tangibles.py -i test_images/program1.jpg 
```
to see all the argumets of the script use:
```
python tangibles.py -h
```

