## Ring Detection in Microscopy Time-Lapse Data

This is a "alpha version" script for detecting ring-like structures in fluorescence microscopy time-lapse recordings. It uses intensity projection and template matching to identify and count ring formations over time.

## How to Use

### 1. Prepare Your Files
- Place your `.tif` movie in a folder.
- Inside the `templates/` folder you should place multiple PNG images of ring-like structures to use as templates for matching.

### 2. Edit the Script

Update the file paths near the bottom of `ring_detection.py`:

```python
template_folder = "templates/*.png"
folder = "path/to/your/movie/folder/"
movie_file = "your_movie.tif"
```

### 3. Run the Script
`python ring_detection.py`

#### Dependencies
`pip install numpy matplotlib scikit-image opencv-python tifffile`
