# -*- coding: utf-8 -*-

"""
Created on Wed Apr 29 15:44:42 2020
@author: pcaldas

"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os

import cv2
from skimage import io
from skimage.feature import match_template, peak_local_max
import tifffile as tiff


def import_templates(temp_dir_files):
    ''' read all the templates in a given folder directory'''
    templates = [io.imread(file) for file in glob.glob(temp_dir_files)]
    return templates

def time_projection(movie_array, step = 30, clip = -1):
    '''computes mean intensity projection acording to step size)
    returns a 2D array with n-step frames of the original movie'''
    
    time_proj_mov = []
    
    for i, frame in enumerate(movie_array[:clip:][:-step]):
  
        mean_proj = np.mean(movie_array[i:i + step], axis = 0)
        #mean_proj = mean_proj / sk.filters.gaussian(mean_proj, sigma = 50)
        time_proj_mov.append(mean_proj)
    
    # each new frame is a projection of the next "step" frames
    return np.array(time_proj_mov)

def save_movie_astif(movie_array, filename = '_time_projection'):
    '''saves movie as tiff file in the same directory of the notebook'''
    with tiff.TiffWriter(filename[:-4] + '.tif', bigtiff = False) as tif:
        for image in range(movie_array.shape[0]):
            tif.save(movie_array[image].astype('float16'), compress = 1)
            
def coord_filtering(points, r):
    """Return a maximal list of elements of points such that no pairs of
    points in the result have distance less than r """
    result = []
    for p in points:
        if all(np.linalg.norm(np.array(p) - np.array(q)) >= r for q in result):
            result.append(p)
    return result

# Main Function - Finds rings in one frame. 
# The analysis works efficiently if the target image is a intensity projection, not the raw data

def ring_search(target_image, template_list, threshold = 0.7, diameter = 8, show_rings = True):
    '''search for rings in the target_image (array) based on the templates in template_list (array)
       threhsold sets the sensibility of the algorithm to find a match, 0.7 by default
       show_rings: if True shows the raw_image with the rings found; diameter sets the diameter of those rings'''
    
    # create empty list to fill with ring coordenates
    all_ring_coord = []

    # loop over different templates
    for template in template_list:
    
        # get template height, width; find center of the template; 
        # use a counter clockwise rotation holding the center for different angles

        (h, w) = template.shape[:2]
        temp_center = (w / 2, h / 2)
        angles = [0, 90, 180, 270]

        for angle in angles:
            
            rotation = cv2.getRotationMatrix2D(temp_center, angle, 1)
            rotated_template = cv2.warpAffine(template, rotation, (h, w))

            # use match_template function from skimage
            # returns the corr. coeff between the image and the template at each position
            
            find_rings = match_template(target_image, rotated_template, pad_input = True)
            
            # to search for the best match we need to search for peaks
            # threshold controls the minimum intensity of peaks, i.e 
            # the minimum corr coeff to consider the ring in this scenario
            
            coordinates = peak_local_max(find_rings, 
                                         num_peaks = 500,
                                         threshold_abs = threshold,
                                         min_distance = diameter,
                                         exclude_border = True, 
                                         indices = True)

            all_ring_coord.append([list(rings) for rings in coordinates])
            
            # pad_input=True matches correspond to the center of the template
            # (otherwise to the top-left corner)
            
    # unpack sublists into a single list of coordenates
    all_ring_coord = [j for i in all_ring_coord for j in i]
    
    # filter overlaping coordenates (i.e they correspond to the same ring)
    all_ring_coord = coord_filtering(all_ring_coord, diameter)
    
    if show_rings == True:
        
        fig, ax = plt.subplots(figsize=(5,5), dpi = 120, constrained_layout = True)

        ax.imshow(target_image, cmap = 'gray')

        for x, y in all_ring_coord:
            circle = plt.Circle((y, x), diameter, color = 'blue', linewidth = 1.2, fill = False)
            ax.add_patch(circle)
            ax.set_title("{} rings found".format(len(all_ring_coord)))

    return all_ring_coord

# Master Function - Apply the previous function to a whole time-lapse. 
# all the previous functions are combined in this one
# the output is a list with all the ring coordenates per frame

def search_rings_movie(filename_dir, templates_dir, threshold = 0.7, time_interval = 2, proj_frames = 30, diameter = 8, clip = -1, step = 1):
    '''search for rings in a time-lapse movie (filename_dir) based on the templates found in template_dir
    a mean intensity projection is first applied. number of frames is set by proj_frames variable, 30 by default
    threhsold sets the sensibility of the algorithm to find a match, 0.7 by default;
    clip: truncate the movie for the analysis, whole movie by default (-1)
    step: analyze only the nth (step) frame for every movie, all frames by default (1)
    time_interval: seconds per frames'''
    
    print('processing the movie (mean intensity projection)')
    time_proj_mov = time_projection(io.imread(filename_dir)[:clip], step = proj_frames) # compute time_projection
    templates = import_templates(templates_dir) # load template images
    
    # create a directory to save image results - this is just a sanity check, we can delete later
    results_dir = 'output_rings_found_'+ str(os.path.basename(filename_dir)[:-4])
    os.makedirs(results_dir, exist_ok = True)

    rings_found = []

    for n, frame in enumerate(time_proj_mov[:clip:step]):
        print('finding rings in frame ' + str(n*step))
        all_rings_in_frame = ring_search(frame, templates, threshold = threshold, show_rings = False)
        rings_found.append([n * step, all_rings_in_frame])

        # save image containing all the rings found - this is just a sanity check, we can delete later
        fig, ax = plt.subplots(figsize=(5,5), dpi = 120, constrained_layout = True)

        ax.imshow(frame, cmap = 'gray')

        for x, y in all_rings_in_frame:
            circle = plt.Circle((y, x), diameter, color = 'blue', linewidth = 1.2, fill = False)
            ax.add_patch(circle)
            ax.set_title("{} rings found".format(len(all_rings_in_frame)))

        plt.savefig(results_dir + '/' + str(n * step * time_interval) + '_sec.png')
        plt.close()

    return rings_found

def plot_results(rings_found):
    '''plot number of rings found over time'''
    
    X,Y = [],[]

    for n_frame,coord in rings_found:
        X.append(n_frame)
        Y.append(len(coord))

    plt.figure(figsize = (4,3), dpi = 120)
    plt.plot(X, Y, '--o', markeredgecolor = 'black')
    plt.xlabel('time (sec)')
    plt.ylabel('# rings found')
    
    return X,Y
    
# RUN THE ANALYSIS

template_folder = "templates/*.png"
folder = "C:\\Users\\comics\\Desktop\\IST Austria\\batir_manuscript\\ring-detection\\1.5 uM FtsZ\\"
wt = 'WT27_00_raw.tif'
exp = "BGS Gaussian_200528_ch7_FtsZ L169R 1.5uM 0.2A_LPO5%_3_Tirf488.tif"

#rings_found = search_rings_movie(folder+wt, template_folder, clip = 500, step = 30)