# Tangibles GSoC repo

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/VasilisPoulos/tangibles-recognition/master)

For more info go to my [gist](https://gist.github.com/VasilisPoulos/5176e80d0f8f4948e0549a58497d3b54).

## How to run

Use the `tang_env.sh` script as:

```shell
source tang_env.sh
```

This will create a python environment in the same directory as the python script 
with all the required libraries installed.

Or, do the same process manually:

```shell
python -m venv tang-env
source tang-env

pip install --upgrade pip
pip install numpy
pip install matplotlib
pip install opencv-python==4.1.2.30
pip install pytesseract
pip install imutils
pip install anytree
sudo apt-get install tesseract-ocr
```

To run the tangibles python script on the test images use:

```shell
python tangibles.py -i test_images/program1.jpg 
python tangibles.py -i test_images/program1.jpg
```

to see all the arguments of the script use:

```shell
python tangibles.py -h
```
