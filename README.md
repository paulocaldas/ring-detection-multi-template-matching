# Ring Detection in Microscopy Time-Lapse Data

This repository contains a Python-based pipeline for detecting ring-like structures in fluorescence microscopy time-lapse recordings. It uses intensity projection and template matching to identify and count ring formations over time.

## How to Use

### 1. Prepare Your Files
- Place your `.tif` movie in a folder.
- Inside the `templates/` folder you should place multiple PNG images of ring-like structures to use as templates for matching.

### 2. Run the Script
python ring_detection.py

#### Dependencies
`pip install numpy matplotlib scikit-image opencv-python tifffile`
