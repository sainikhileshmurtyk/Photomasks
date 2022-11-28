## Importing the necessary Packages
# System packages
import os
import csv

# Science Packages
import numpy as np
import scipy as sc

# CAD Packages
import gdstk

## Setting up the Units
# default size unit is microns, so define handy shortcuts for other units
um = 1
mm = 1e3
cm = 1e4


def convert_to_pygds_dimensions(mask_positions, mask_size):
    vert_pos = np.copy(mask_positions[0,...])
    horz_pos = np.copy(mask_positions[1,...])
    
    x_pos = horz_pos
    y_pos = mask_size-vert_pos
    
    mask_positions[0,...] = x_pos
    mask_positions[1,...] = y_pos
    
    return mask_positions

# Utilities
def convert_rect(size, center):
    coord1 = (center[0] - (size[0]/2), center[1] - (size[1]/2))
    coord2 = (center[0] + (size[0]/2), center[1] + (size[1]/2))
    return coord1, coord2

def invert_coords(x_arr, y_arr):
    """
    invert_coords(x_arr, y_arr)

    Inputs
    ------
    x_arr: np.array, 
    """
    tmp_x = y_arr
    tmp_y = x_arr
    return tmp_x, tmp_y


def convert_coords(dims, x_arr, y_arr):
    if type(dims) == float:
        tmp_y = dims - y_arr
        tmp_x = x_arr
    elif type(dims) == list:
        tmp_y = dims[0] - y_arr
        tmp_x = x_arr
    return tmp_x, tmp_y


def shift_to_center(dims, x_arr, y_arr):
    if type(dims) == float:
        tmp_x = x_arr - (dims/ 2)
        tmp_y = y_arr - (dims/ 2)
    elif type(dims) == list:
        tmp_x = x_arr - (dims[0]/ 2)
        tmp_y = y_arr - (dims[1]/ 2)
    return tmp_x, tmp_y


def shift_to_zero(dims, x_arr, y_arr):
    if type(dims) == float:
        tmp_x = x_arr + (dims/ 2)
        tmp_y = y_arr + (dims/ 2)
    elif type(dims) == list:
        tmp_x = x_arr + (dims[0]/ 2)
        tmp_y = y_arr + (dims[1]/ 2)
    return tmp_x, tmp_y

def merge_centers(led_real_pos, pat_center, dims, invert = True, index = 0):
    if invert == True:
        center_led = [(led_real_pos['x'][index, dims[1]-1] + led_real_pos['x'][index,0])/2, (led_real_pos['y'][0,index] + led_real_pos['y'][dims[0]-1, index])/2]
    elif invert == False:
        center_led = [(led_real_pos['x'][index, dims[1] -1] + led_real_pos['x'][index,0])/2, (led_real_pos['y'][dims[0]-1, index] + led_real_pos['y'][0, index])/2]
    shift = [pat_center[0] - center_led[0] , pat_center[1] - center_led[1]] 
    return shift