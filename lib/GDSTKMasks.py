# System packages
import copy
# Data Packages
import json

# CAD Packages
import gdstk

# Science Packages
import numpy as np
import scipy as sc

import lib.dither_utils as dit
import lib.GDSTKUtils as utils
from lib.gds_photomask_library import pinhole_pat,rect_border,chequerboard_pat,single_chequerboard_pat,calib_mask,dithered_pat,position_pattern, convert_to_pygds_dimensions

# Setting units
# default size unit is microns, so define handy shortcuts for other units
um = 1
mm = 1e3
cm = 1e4

# Defining Marks
def rect_border(center, size, thickness = 750*um):
    """
    rect_border(center, size, thickness = 750*um)

    Arguments
    =========
    center:
        list 1-by-2, The center of the border
    size: 
        list 1-by-2, Dimensions of the border
    thickness: 
        float, thickness of the border 
    """
    # Converting the coordinates of the rectangle from the preferred coordinates to the GDSTK coordinates
    coords_big = utils.convert_rect(size, center)
    coords_small = utils.convert_rect([num - thickness for num in size], center)

    # Creating the big rectangle
    rect_big = gdstk.rectangle(coords_big[0], coords_big[1])
    
    # Creating the smaller rectangle
    rect_small = gdstk.rectangle(coords_small[0], coords_small[1])
    
    # Creating the pattern by subtracting the bigger rectangle and the smaller rectangle
    pattern = gdstk.boolean(rect_big, rect_small, "not")
    
    return pattern

def pinhole_array(mask, center, led_real_pos, size, title):
    """
    pinhole_array(mask, pos_x, pos_y, dims)

    Arguments
    =========
    mask: 
        GDSTK mask object, mask to which the chequerboard needs to be added
    center:
        list 1-by-2, Coordinates of the center position of the pattern
    led_real_pos:
        dict, Dictionary containing the keys 'x' and 'y' denoting the x and y positions of the LEDs
    size:
        tuple, x and y size of the pinhole in real units
    """
    # Creating a slide border
    slide_border = rect_border([center[0], center[1]], [30.25*mm, 30.25*mm], 250*um)
    mask.add(*slide_border)

    # Shifting the coordinate system to the center of the mask
    pos_x, pos_y = utils.shift_to_center([25*mm, 25*mm], led_real_pos['x'], led_real_pos['y'])

    # Number of LEDS
    dims = led_real_pos['x'].shape

    # Adding the center to shift the mask to the correct position
    pos_x, pos_y = pos_x + center[0], pos_y + center[1]

    # Looping over the number of elements and generating the masks
    for idx1 in range(np.shape(pos_x.flatten())[0]):
        pinhole = gdstk.rectangle(*utils.convert_rect([size[0], size[1]], [pos_x.flatten()[idx1], pos_y.flatten()[idx1]]))
        mask.add(pinhole)

    # Adding the description text to the mask
    text_pos = (center[0] - 14.5*mm, center[1] - 14.5*mm)
    desc = gdstk.text(title, 0.75*mm, text_pos)
    mask.add(*desc)

    return mask

def pinhole_array_shifted(mask, center, led_real_pos, size, magnification, title):
    """
    pinhole_array_shifted(mask, pos_x, pos_y, dims)

    Arguments
    =========
    mask: 
        GDSTK mask object, mask to which the chequerboard needs to be added
    center:
        list 1-by-2, Coordinates of the center position of the pattern
    led_real_pos:
        dict, Dictionary containing the keys 'x' and 'y' denoting the x and y positions of the LEDs
    size:
        tuple, x and y size of the pinhole in real units
    magnification:
        int, Magnification of the next stage of the device
    """
    # Creating a slide border
    slide_border = rect_border([center[0], center[1]], [30.25*mm, 30.25*mm], 250*um)
    mask.add(*slide_border)

    # Number of LEDS
    dims = led_real_pos['x'].shape

    # Finding the center of the LED
    center_led = [(led_real_pos['x'][dims[0]-1, dims[1]-1] + led_real_pos['x'][0,0])/2, (led_real_pos['y'][0,0] + led_real_pos['y'][dims[0]-1, dims[1]-1])/2]

    # Shifting the coordinate system to the center of the mask
    pos_x, pos_y = utils.shift_to_center([25*mm, 25*mm], led_real_pos['x'], led_real_pos['y'])

    # Calculating the shifts of the position
    shift_led = [-(1/magnification * (led_real_pos['x'] - center_led[0])), -(1/magnification * (led_real_pos['y'] - center_led[1]))]

    # Shifting the matrix according to the calculated shift positions
    pos_x, pos_y = pos_x + shift_led[0], pos_y + shift_led[1]

    # Adding the center to shift the mask to the correct position
    pos_x, pos_y = pos_x + center[0], pos_y + center[1]

    # Looping over the number of elements and generating the masks
    for idx1 in range(np.shape(pos_x.flatten())[0]):
        pinhole = gdstk.rectangle(*utils.convert_rect([size[0], size[1]], [pos_x.flatten()[idx1], pos_y.flatten()[idx1]]))
        mask.add(pinhole)

    # Adding the description text to the mask
    text_pos = (center[0] - 14.5*mm, center[1] - 14.5*mm)
    desc = gdstk.text(title, 0.75*mm, text_pos)
    mask.add(*desc)

    return mask

def chequerboard_array(mask, pat_center, magnification, led_center, pd_center, title, box_size = [150*um, 150*um]):
    """
    chequerboard_array(mask, pat_center, magnification, led_real_pos, pd_real_pos, title, box_size = [150*um, 150*um])

    Arguments
    =========
    mask: 
        GDSTK mask object, The mask that needs to be written to the gds file
    pat_center: 
        list 1-by-2, The center of the pattern that needs to be written
    magnification: 
        int, magnification between two subsequent stages
    led_real_pos: 
        np.ndarray, The center positions of the LEDs of the previous layer
    pd_real_pos: 
        np.ndarray, The center positions to the PDs of the subsequent layer
    title: 
        string, The title of the mask to be made 
    box_size: 
        list 1-by-2, The box size of the individual chequerboard pattern
    """
    # Creating a slide border
    slide_border = rect_border([pat_center[0], pat_center[1]], [30.25*mm, 30.25*mm], 250*um)
    mask.add(*slide_border)
    
    # Copying the PD and LED positions
    led_real_pos = copy.deepcopy(led_center)
    pd_real_pos = copy.deepcopy(pd_center)
    
    # Shifting the positions of the photodiodes of the subsequent layer to the center for calculating shifts
    pd_real_pos['x'], pd_real_pos['y'] = utils.shift_to_center([25*mm, 25*mm], pd_real_pos['x'], pd_real_pos['y'])

    # The number of LED's in the previous layer
    dims = led_real_pos['x'].shape
    
    # Finding the center of the LED
    center_led = [(led_real_pos['x'][dims[0]-1, dims[1]-1] + led_real_pos['x'][0,0])/2, (led_real_pos['y'][0,0] + led_real_pos['y'][dims[0]-1, dims[1]-1])/2]
    
    # Finding the shift to position the chequerboard pattern at the center of the mask
    shift = utils.merge_centers(led_real_pos, pat_center, dims=dims)

    # Shifting the mask in fornt of the LEDs so tha tthey are coincident on the photodiodes
    shift_led = [-(1/magnification * (led_real_pos['x'] - center_led[0])) + shift[0], -(1/magnification * (led_real_pos['y'] - center_led[1])) + shift[1]]
    center_pos = [shift_led[0] + led_real_pos['x'], shift_led[1] + led_real_pos['y']]

    # Calculating the center positions of the chequerboard
    center_cb = [-(1/magnification) * pd_real_pos['x'], -(1/magnification) * pd_real_pos['y']]

    # Matrix that represents the presence or absence of a chequerboard pattern to create alternating patterns
    tmat = sc.linalg.toeplitz(np.mod(np.arange(1,11,1), 2))

    # Looping through the LED positions
    for li in range(0, led_real_pos['x'].shape[0]):
        for lj in range(0, led_real_pos['y'].shape[1]):

            # Diagnostics to check the positioning of the LED box
            # border = rect_border([center_pos[0][li,lj], center_pos[1][li,lj]], [2.25*mm, 2.25*mm], 250*um)
            # mask.add(*border)

            # Adding alignment markers between the spots
            marker_11 = gdstk.rectangle(*utils.convert_rect([200*um, 200*um], [shift_led[0][li,lj] + led_real_pos['x'][li,lj] - 1.5*mm, shift_led[1][li,lj] + led_real_pos['y'][li,lj] + 1.5*mm])) # top-left
            marker_21 = gdstk.rectangle(*utils.convert_rect([200*um, 200*um], [shift_led[0][li,lj] + led_real_pos['x'][li,lj] - 1.5*mm, shift_led[1][li,lj] + led_real_pos['y'][li,lj] - 1.5*mm])) # bottom-left
            marker_12 = gdstk.rectangle(*utils.convert_rect([200*um, 200*um], [shift_led[0][li,lj] + led_real_pos['x'][li,lj] + 1.5*mm, shift_led[1][li,lj] + led_real_pos['y'][li,lj] + 1.5*mm])) # top-right
            marker_22 = gdstk.rectangle(*utils.convert_rect([200*um, 200*um], [shift_led[0][li,lj] + led_real_pos['x'][li,lj] + 1.5*mm, shift_led[1][li,lj] + led_real_pos['y'][li,lj] - 1.5*mm])) # bottom-right
            mask.add(marker_11, marker_12, marker_21, marker_22)
 
            # Shifting the center of each of the chequerboard position matrix
            center_cb_shift_x, center_cb_shift_y = center_cb[0] + center_pos[0][li,lj], center_cb[1] + center_pos[1][li,lj]
            for idx1 in range(0, center_cb_shift_x.shape[0]):
                for idx2 in range(0, center_cb_shift_x.shape[1]):
                    if tmat[idx1, idx2] == 1:
                        # Creating a new rectangle a tthe position of the individual chequeboard pattern
                        rect = gdstk.rectangle(*utils.convert_rect(box_size, [center_cb_shift_x[idx1, idx2], center_cb_shift_y[idx1, idx2]]))
                        mask.add(rect)

    # Creating and adding the text description at the bottom
    text_pos = (pat_center[0] - 14.5*mm, pat_center[1] - 14.5*mm)
    desc = gdstk.text(title, 0.75*mm, text_pos)
    mask.add(*desc)

    # Deleting all the used variables
    del slide_border, pat_center, pd_real_pos, led_real_pos, center_led, shift, dims, magnification, center_pos, shift_led, tmat, marker_11, marker_21, marker_12, marker_22, center_cb_shift_x, center_cb_shift_y, text_pos, desc,  
    
    return mask


def generate_weight_mask(masks, center, net, led_center, pd_center, magnification, title, dither_size = 200):
    """
    generate_weight_mask(masks, key, json_file, center, net, led_center, pd_center):

    Arguments
    =========
    masks:
        GDSTK mask object, The mask to which patterns need to be added
    center:
        list 1-by-2, center of the pattern 
    net:
        np.ndarray, Matrix of matrices of weights
    led_center:
        np.ndarray, Center of the LEDs of the previous layer
    pd_center:
        np.ndarray, Center of the PDs of the previous layer
    magnification:
        int, Magnification between the LEDs of the previous layer and the photodiode of the subsequent layer
    title: 
        string, Title of the photomask at the bottom
    dither_size:
        float, Size of the smaller dither block
    """
    # Creating the cut border 0f 30mm by 20mm around the mask
    slide_border = rect_border([center[0], center[1]], [30.25*mm, 30.25*mm], 250*um)
    masks.add(*slide_border)

    # Selecting the type of the mask
    sel_cell = copy.deepcopy(net)

    # Smaller block sizes
    inner_dims = sel_cell[0,0].shape

    # Setting useful values
    dims = sel_cell.shape

    # Setting the sart, step and end values of the ideal positions
    ideal_start_pos, ideal_end_pos, ideal_sep = 1.25*mm, 25*mm, 2.5*mm

    # Initializing an empty dictionary to hold ideal positions
    pd_ideal_pos = {}
    pd_real_pos = {}
    led_real_pos = {}

    # Generating a meshgrid of values of idela positions
    pd_ideal_arr = np.arange(ideal_start_pos, ideal_end_pos, ideal_sep)

    # Creating a meshgrid of positions
    pd_ideal_pos['x'], pd_ideal_pos['y'] =  np.meshgrid(pd_ideal_arr, pd_ideal_arr)
    
    # Converting the coordinates to GDSTK frame of reference
    pd_ideal_pos['x'], pd_ideal_pos['y'] = utils.convert_coords(25*mm, pd_ideal_pos['x'], pd_ideal_pos['y'])
    
    # Shifting the coordinates to the center
    pd_ideal_pos['x'], pd_ideal_pos['y'] = utils.shift_to_center(25*mm, pd_ideal_pos['x'], pd_ideal_pos['y'])


    # Assigning the created variables
    pd_real_pos = copy.deepcopy(pd_center)
    pd_real_pos['x'], pd_real_pos['y'] = utils.shift_to_center(25*mm, pd_real_pos['x'], pd_real_pos['y'])
    led_real_pos = copy.deepcopy(led_center)
    
    # Finding center position of the matrix
    center_led = [(led_real_pos['x'][dims[0]-1, dims[1]-1] + led_real_pos['x'][0,0])/2, (led_real_pos['y'][0,0] + led_real_pos['y'][dims[0]-1, dims[1]-1])/2]

    # Finding shifts
    shift = utils.merge_centers(led_real_pos, center, dims=dims)

    for li in range(0, dims[0]):
        for lj in range(0, dims[1]):

            # Initializing an empty dictionary to hold coordinates and dimensions of the dithered values
            dither_mat = {}
            
            # Dithering the values in the 8 by 8 matrix
            dither_mat['y'], dither_mat['x'], dither_mat['dim'] = dit.dither_16x(sel_cell[li,lj].flatten(), ((dither_size/16) * 10**-3)) # Because the output from the dither_16x returns y dimension first
            dither_mat['x'], dither_mat['y'], dither_mat['dim'] = np.array(dither_mat['x'], dtype=object)*mm, np.array(dither_mat['y'], dtype=object)*mm, np.array(dither_mat['dim'], dtype=object)*mm
            
            # Solution for the problem with recasting a matrix of empty arrays. This creates a new matrix of empty arrays of suitable size instead of recasting
            if dither_mat['x'].size == 0:
                empty_mat = np.empty(inner_dims, dtype=object)
                for li_loop1 in range(0, empty_mat.shape[0]):
                    for lj_loop1 in range(0, empty_mat.shape[1]):
                        empty_mat[li_loop1,lj_loop1] = np.array([])
                dither_mat['x'], dither_mat['y'], dither_mat['dim'] = empty_mat, empty_mat, empty_mat
            else:
                # Else, do the reshaping
                dither_mat['x'], dither_mat['y'], dither_mat['dim'] = dither_mat['x'].reshape(inner_dims), dither_mat['y'].reshape(inner_dims), dither_mat['dim'].reshape(inner_dims)
            
            # Shifting the positions of the 10 by 10 weight matrices based on the LED positions and the also shifted for all of the patterns to be coincidental on the next layer
            shift_led = [-(1/magnification * (led_real_pos['x'][li,lj] - center_led[0])) + shift[0], -(1/magnification * (led_real_pos['y'][li,lj] - center_led[1])) + shift[1]]
            dither_mat['x'], dither_mat['y'], dither_mat['dim'] = dither_mat['x'] + led_real_pos['x'][li,lj] + shift_led[0], dither_mat['y'] + led_real_pos['y'][li,lj] + shift_led[1], dither_mat['dim']

            # Adding alignment markers between the spots
            marker_11 = gdstk.rectangle(*utils.convert_rect([200*um, 200*um], [shift_led[0] + led_real_pos['x'][li,lj] - 1.5*mm, shift_led[1] + led_real_pos['y'][li,lj] + 1.5*mm])) # top-left
            marker_21 = gdstk.rectangle(*utils.convert_rect([200*um, 200*um], [shift_led[0] + led_real_pos['x'][li,lj] - 1.5*mm, shift_led[1] + led_real_pos['y'][li,lj] - 1.5*mm])) # bottom-left
            marker_12 = gdstk.rectangle(*utils.convert_rect([200*um, 200*um], [shift_led[0] + led_real_pos['x'][li,lj] + 1.5*mm, shift_led[1] + led_real_pos['y'][li,lj] + 1.5*mm])) # top-right
            marker_22 = gdstk.rectangle(*utils.convert_rect([200*um, 200*um], [shift_led[0] + led_real_pos['x'][li,lj] + 1.5*mm, shift_led[1] + led_real_pos['y'][li,lj] - 1.5*mm])) # bottom-right
            masks.add(marker_11, marker_12, marker_21, marker_22)

            # Looping within the 10 by 10 matrix 
            for idx1 in range(0, dither_mat['x'].shape[0]):
                for idx2 in range(0, dither_mat['x'].shape[1]):
                    sel_mat = {}
                    sel_mat['x'], sel_mat['y'], sel_mat['dim'] = dither_mat['x'][idx1, idx2], dither_mat['y'][idx1, idx2], dither_mat['dim'][idx1, idx2]
                    if sel_mat['x'].size == 0 or sel_mat['y'].size == 0:
                        # Skipping if the matrix is empty. Happens for 8 by 8 row for input
                        continue

                    # Shifting the matrix according to the real positions of the photodiodes so the shifts are accounted for
                    sel_mat['x'], sel_mat['y'], sel_mat['dim'] = sel_mat['x'] - ((1/magnification) * pd_real_pos['x'][idx1, idx2]), sel_mat['y'] - ((1/magnification) * pd_real_pos['y'][idx1, idx2]), sel_mat['dim']
                    
                    # Placing rectangles at appropriate positions by looping through the dither blocks
                    for idx3 in range(0, sel_mat['x'].shape[0]):
                        pattern = gdstk.rectangle(*utils.convert_rect([sel_mat['dim'][idx3], sel_mat['dim'][idx3]], [sel_mat['x'][idx3], sel_mat['y'][idx3]]))
                        masks.add(pattern)
            
    # Adding the description text to the mask
    text_pos = (center[0] - 14.5*mm, center[1] - 14.5*mm)
    desc = gdstk.text(title, 0.75*mm, text_pos)
    masks.add(*desc)

    return masks

def generate_weight_mask_from_json(masks, key, json_file, center, net, led_center, pd_center):
    """
    generate_weight_mask_from_json(masks, key, json_file, center, net, led_center, pd_center):

    Arguments
    =========
    masks:
        GDSTK mask object, The mask to which patterns need to be added
    key:
        int, denotes the weight layer of the mask
    json_file:
        os.path, Path of the json file which needs to be opened
    center:
        list 1-by-2, center of the pattern 
    net:
        np.ndarray, Matrix of matrices of weights
    led_center:
        np.ndarray, Center of the LEDs of the previous layer
    pd_center:
        np.ndarray, Center of the PDs of the previous layer
    """
    # Reading from the JSON file
    with open(json_file, 'r') as openfile:
        param_dict = json.load(openfile)
    params = param_dict[key]
    magnification = params['magnification']
    title = params["title"]
    dither_size = params["dither_size"]

    # Passing the variables on to the fucntion to generate the mask
    masks = generate_weight_mask(masks, center, net, led_center, pd_center, magnification, title, dither_size)

    return masks

def generate_weight_mask_from_dict(masks, dictionary, center, net, led_center, pd_center):
    """
    generate_weight_mask_from_dict(masks, key, json_file, center, net, led_center, pd_center):

    Arguments
    =========
    masks:
        GDSTK mask object, The mask to which patterns need to be added
    dictionary:
        dict, Dictionary containing the variables about the mask
    center:
        list 1-by-2, center of the pattern 
    net:
        np.ndarray, Matrix of matrices of weights
    led_center:
        np.ndarray, Center of the LEDs of the previous layer
    pd_center:
        np.ndarray, Center of the PDs of the previous layer
    """
    # Reading from the dictionary
    params = dictionary
    magnification = params['magnification']
    title = params["title"]
    dither_size = params["dither_size"]

    # Passing the variables on to the fucntion to generate the mask
    masks = generate_weight_mask(masks, center, net, led_center, pd_center, magnification, title, dither_size)

    return masks

def sel_mask(mask, mask_dict, center):
    """
    sel_mask(mask, mask_dict, center)

    Arguments
    =========
    mask:
        GDSTK mask object, Mask on which the postions need to be overlaid
    mask_dict:
        dict, Dictionary containing the different mask parameters
    center:
        tuple, center for the mask pattern to be created
    """
    if mask_dict['type'] == "ph_array":
        pos_dict = mask_dict["led_center"]
        dims = mask_dict["dims"]
        title = mask_dict["title"]
        mask = pinhole_array(mask, center, pos_dict, dims, title)

    elif mask_dict["type"] == "ph_array_shifted":
        pos_dict = mask_dict["led_center"]
        dims = mask_dict["dims"]
        magnification = mask_dict["magnification"]
        title = mask_dict["title"]
        mask = pinhole_array_shifted(mask, center, pos_dict, dims, magnification, title)

    elif mask_dict['type'] == "cb_array":
        magnification = mask_dict["magnification"]
        led_real_pos = mask_dict["led_center"]
        pd_real_pos = mask_dict["pd_center"]
        title = mask_dict["title"]
        box_size = mask_dict["box_size"]
        mask = chequerboard_array(mask, center, magnification, led_real_pos, pd_real_pos, title, box_size)

    elif mask_dict['type'] == "weight_mask":
        net = mask_dict["net"]
        led_center = mask_dict["led_center"]
        pd_center = mask_dict["pd_center"]
        mask = generate_weight_mask_from_dict(mask, mask_dict, center, net, led_center, pd_center)
    else:
        raise Exception("Please enter a valid type of dictionary OR the dictionary entered does not have a valid key")
        
    return mask