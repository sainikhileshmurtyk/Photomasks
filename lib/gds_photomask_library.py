# Code from Nikhilesh - utility functions from [220422] For Order jupyter notebook

import gdspy
import numpy as np
import os
import gdstk
#from IPython.display import clear_output

# default size unit is microns, so define handy shortcuts for other units
um = 1
mm = 1e3
cm = 1e4


# %%

# Defining Marks
def hole_array(start_pos = [0,0], block_size = [30*mm, 30*mm], num_holes = [5,10], dia_circle = 225*um, circle_sep = [2.5*mm, 5*mm], init_pos = [3.75*mm, 5*mm]):
    rad_circle = dia_circle/2
    Rectangle = gdspy.Rectangle((0, 0), (block_size[0], block_size[1]))
    sub_rect = Rectangle
    count = 1
    x_pos = init_pos[0]
    y_pos = init_pos[1]
    pat_list = []
    for li in range(1, num_holes[0]+1):
        x_pos = init_pos[0]
        for lj in range(1, num_holes[1]+1):
            print("Hole Array \n Index: ", str(count), "/", str(num_holes[0] * num_holes[1]))
            tmp_circle = gdspy.Round((x_pos, y_pos), rad_circle)
            sub_rect = gdspy.boolean(sub_rect, tmp_circle, operation = "not")
            x_pos += circle_sep[0]
            count += 1
            pat_list.append(tmp_circle)
#            clear_output(wait=True)
        y_pos += circle_sep[1]
    return pat_list

def circ_hole_array(x_pos_array, y_pos_array, dia):
    pat_list = []
    for li in range(0, len(x_pos_array)):
        pat_list.append(gdspy.Round([x_pos_array[li], y_pos_array[li]], dia/2))
    return pat_list
        

def rect_hole_array(x_pos_array, y_pos_array, dim):
    pat_list = []
    for li in range(0, len(x_pos_array)):
        rect_coords = convert_rect(dim, [x_pos_array[li], y_pos_array[li]])
        hole = gdspy.Rectangle(rect_coords[0], rect_coords[1])
        pat_list.append(hole)
    return pat_list


def cb_array(start_pos, dim_arr, size_cboard, separation):
    if type(separation) == int:
        separation_x = separation
        separation_y = separation
    else:
        separation_x = separation[0]
        separation_y = separation[1]
    pos_x = start_pos[0]
    pos_y = start_pos[1]
    idx = 1
    lists = []
    for li in range(1, dim_arr[0]+1):
        pos_x = start_pos[0]
        for lj in range(1, dim_arr[1]+1):
            rect_coord = convert_rect([size_cboard, size_cboard], [pos_x, pos_y])
            pattern = gdspy.Rectangle(rect_coord[0], rect_coord[1])
            pos_x += separation_x
            idx += 1
            lists.append(pattern)
        pos_y += separation_y
    return lists

def rand_cb_array(start_pos, dim_arr, size_cboard, separation, trans_mat, factor):
    pos_x = start_pos[0]
    pos_y = start_pos[1]
    idx = 1
    idx2 = 0
    lists = []
    for li in range(1, dim_arr[0]+1):
        pos_x = start_pos[0]
        for lj in range(1, dim_arr[1]+1):
            rect_coord = convert_rect([size_cboard, size_cboard], [pos_x, pos_y])
            if trans_mat[idx2] > factor:
                pattern = gdspy.Rectangle(rect_coord[0], rect_coord[1])
                lists.append(pattern)
            pos_x += separation
            idx += 1
            idx2 += 1     
        pos_y += separation
    return lists
    
def alignment_cross(center, size):
    cross1_coords = convert_rect([size[0], size[1]], center)
    cross2_coords = convert_rect([size[1], size[0]], center)
    cross1 = gdspy.Rectangle(cross1_coords[0], cross1_coords[1])
    cross2 = gdspy.Rectangle(cross2_coords[0], cross2_coords[1])
    cross = gdspy.boolean(cross1, cross2, operation="or")
    return cross

def rect_border(center, size, thickness = 750*um):
#    a = 1
    coords_big = convert_rect(size, center)
    rect_big = gdspy.Rectangle(coords_big[0], coords_big[1])
    coords_small = convert_rect([num - thickness for num in size], center)
    rect_small = gdspy.Rectangle(coords_small[0], coords_small[1])
    pattern = gdspy.boolean(rect_big, rect_small, "not")
    return pattern

# Device
def pinhole_array(block_size = [30*mm, 30*mm], num_holes = [5,10], dia_circle = 225*um, circle_sep = [2.5*mm, 5*mm], init_pos = [3.75*mm, 5*mm], cross_size = [750*um, 150*um], block_edge = 2*mm):
    holes = hole_array(block_size, num_holes, dia_circle, circle_sep, init_pos)
    cross_center_dim1 = block_edge
    cross_center_dim2 = block_size[1] - block_edge
    cross1_cent = [cross_center_dim1, cross_center_dim1]
    cross2_cent = [cross_center_dim1, cross_center_dim2]
    cross3_cent = [cross_center_dim2, cross_center_dim1]
    cross4_cent = [cross_center_dim2, cross_center_dim2]
    cross = alignment_cross(cross1_cent, cross_size)
    holes = gdspy.boolean(holes,cross, "not")
    cross = alignment_cross(cross2_cent, cross_size)
    holes = gdspy.boolean(holes,cross, "not")
    cross = alignment_cross(cross3_cent, cross_size)
    holes = gdspy.boolean(holes,cross, "not")
    cross = alignment_cross(cross4_cent, cross_size)
    holes = gdspy.boolean(holes,cross, "not")
    return holes

def convert_to_pygds_dimensions(mask_positions,mask_size):
    vert_pos = np.copy(mask_positions[0,...])
    horz_pos = np.copy(mask_positions[1,...])
    
    x_pos = horz_pos
    y_pos = mask_size-vert_pos
    
    mask_positions[0,...] = x_pos
    mask_positions[1,...] = y_pos
    
    return mask_positions

# High thoroughput readers
def position_func(mask_positions, translate_coords):

    # Reading into a numpy array
    pat_list = []
    for idx in range(mask_positions.shape[1]):
        x_pos = mask_positions[0,idx]*mm+translate_coords[0]
        y_pos = mask_positions[1,idx]*mm+translate_coords[1]
        dither_size = mask_positions[2,idx]*mm
        rect_coord = convert_rect([dither_size, dither_size], [x_pos, y_pos])
        dither_rect = gdspy.Rectangle(rect_coord[0], rect_coord[1])
        pat_list.append(dither_rect)
    return pat_list


# High thoroughput readers
def dither_func(coord_path, num, position_mat_x, position_mat_y, dither_size, translate_coords):
    num_elems = len([name for name in os.listdir(coord_path) if os.path.isfile(os.path.join(coord_path, name))])

    # Reading into a numpy array
    coord_cell = np.empty(num_elems, dtype=object)
    for idx in range(1, num_elems+1):
        coord_cell[idx-1] = np.loadtxt(os.path.join(coord_path, "mask%01d_%02d.csv" % (num, idx)), delimiter=",", dtype=float)
    x, y = np.meshgrid(position_mat_x, position_mat_y)
    elem_idx = 0
    pat_list = []
    for idx_x in range(0, len(position_mat_x)):
        for idx_y in range(0, len(position_mat_y)):
            x_add = x[idx_y][idx_x] - translate_coords[0]
            y_add = y[idx_y][idx_x] - translate_coords[1]
            for idx2 in range(0, np.shape(coord_cell[elem_idx])[0]):
                x_pos = x_add + (coord_cell[elem_idx][idx2][0] * mm)
                y_pos = y_add + (coord_cell[elem_idx][idx2][1] * mm)
                rect_coord = convert_rect([dither_size, dither_size], [x_pos, y_pos])
                dither_rect = gdspy.Rectangle(rect_coord[0], rect_coord[1])
                pat_list.append(dither_rect)
                # mask.add(dither_rect)
            elem_idx += 1
    return pat_list

def set_rect(coord_path, num, position_mat_x, position_mat_y, rect_size, translate_coords):
    # num_elems = len([name for name in os.listdir(coord_path) if os.path.isfile(os.path.join(coord_path, name))])
    num_elems = 50
    # Reading into a numpy array
    coord_cell = np.empty(num_elems, dtype=object)
    for idx in range(1, num_elems+1):
        coord_cell[idx-1] = np.loadtxt(os.path.join(coord_path, "mask_cal_01.csv"), delimiter=",", dtype=float)
    x, y = np.meshgrid(position_mat_x, position_mat_y)
    elem_idx = 0
    pat_list = []
    for idx_x in range(0, len(position_mat_x)):
        for idx_y in range(0, len(position_mat_y)):
            x_add = x[idx_y][idx_x] - translate_coords[0]
            y_add = y[idx_y][idx_x] - translate_coords[1]
            for idx2 in range(0, np.shape(coord_cell[elem_idx])[0]):
                x_pos = x_add + (coord_cell[elem_idx][idx2][0] * mm)
                y_pos = y_add + (coord_cell[elem_idx][idx2][1] * mm)
                rect_coord = convert_rect([rect_size, rect_size], [x_pos, y_pos])
                dither_rect = gdspy.Rectangle(rect_coord[0], rect_coord[1])
                pat_list.append(dither_rect)
                # mask.add(dither_rect)
            elem_idx += 1
    return pat_list

# Utilities
def convert_rect(size, center):
    coord1 = (center[0] - (size[0]/2), center[1] - (size[1]/2))
    coord2 = (center[0] + (size[0]/2), center[1] + (size[1]/2))
    return coord1, coord2

#def create_arr(pattern, trans_mat, rep_mat):
#    for lx in range(1, rep_mat[0]+1):
#        for ly in range(1, rep_mat[1]+1):
#            for li in range(0, len(pattern)):
#                new_list.append(gdspy.copy(pattern[li], dx = trans_mat[0], dy = trans_mat[1]))
#    return new_list

def chequerboard_pat(mask, position_mat, shift_mat, separation, num_rep, size_cboard, dim_arr, text):
    position_mat_x = position_mat[0]
    position_mat_y = position_mat[1]

    cross_x = [3*mm, 27*mm, 3*mm, 27*mm]
    cross_y = [3*mm, 3*mm, 27*mm, 27*mm]

    text_location_x = 4*mm
    text_location_y = 2.85*mm

    # Shifts
    shift_x = shift_mat[0]
    shift_y = shift_mat[1]
    position_mat_x = [x + shift_x for x in position_mat_x]
    position_mat_y = [y + shift_y for y in position_mat_y]
    
    cross_x = [x + shift_x for x in cross_x]
    cross_y = [y + shift_y for y in cross_y]
    
    text_location_x = text_location_x + shift_x
    text_location_y = text_location_y + shift_y
    
    if num_rep == [10,5]:
        sub_1 = 922.5*um
        sub_2 = 717.5*um
    elif num_rep == [3,4]:
        sub_1 = 1004.5*um
        sub_2 = 799.5*um
    elif num_rep == [7,7]:
        sub_1 = 1350*um
        sub_2 = 1050*um
    else:
        sub_1 = 1004.5*um
        sub_2 = 799.5*um

    # Creating a Single pattern
    for li in range(0, num_rep[0]):
        for lj in range(0, num_rep[1]):
            start_pat1 = [position_mat_x[li] - sub_1, position_mat_y[lj] - sub_2]
            start_pat2 = [position_mat_x[li] - sub_2, position_mat_y[lj] - sub_1]
            pat1 = cb_array(start_pat1, dim_arr, size_cboard, separation=separation)
            pat2 = cb_array(start_pat2, dim_arr, size_cboard, separation=separation)
            mask.add(pat1)
            mask.add(pat2)

    # Alignment Cross
    for idx in range(0, len(cross_x)):
        a_cross = alignment_cross([cross_x[idx], cross_y[idx]], [200*um, 50*um])
        mask.add(a_cross)

    # Text
    text_element = gdspy.Text(text, size=300, position=[text_location_x, text_location_y])
    mask.add(text_element)

    return None
def single_chequerboard_pat(mask, position_mat, shift_mat, separation_mat, num_rep, size_cboard, dim_arr, text):
    position_mat_x = position_mat[0]
    position_mat_y = position_mat[1]

    cross_x = [3*mm, 27*mm, 3*mm, 27*mm]
    cross_y = [3*mm, 3*mm, 27*mm, 27*mm]

    text_location_x = 4*mm
    text_location_y = 2.85*mm

    # Shifts
    shift_x = shift_mat[0]
    shift_y = shift_mat[1]
    position_mat_x = [x + shift_x for x in position_mat_x]
    position_mat_y = [y + shift_y for y in position_mat_y]
    
    cross_x = [x + shift_x for x in cross_x]
    cross_y = [y + shift_y for y in cross_y]
    
    text_location_x = text_location_x + shift_x
    text_location_y = text_location_y + shift_y

    # Creating a Single pattern
    for li in range(0, num_rep[0]):
        for lj in range(0, num_rep[1]):
            start_pat = [position_mat_x[li] - 717.5*um + 102.5*um, position_mat_y[lj] - 922.5*um - 154.75*um]
            pat = cb_array(start_pat, dim_arr, size_cboard, separation=separation_mat)
            mask.add(pat)

    # Alignment Cross
    for idx in range(0, len(cross_x)):
        a_cross = alignment_cross([cross_x[idx], cross_y[idx]], [200*um, 50*um])
        mask.add(a_cross)

    # Text
    text_element = gdspy.Text(text, size=300, position=[text_location_x, text_location_y])
    mask.add(text_element)

def pinhole_pat(mask, position_mat, shift_mat, dim):
    # Setting parameters
    position_mat_x = position_mat[0]
    position_mat_y = position_mat[1]

    cross_x = [3*mm, 27*mm, 3*mm, 27*mm]
    cross_y = [3*mm, 3*mm, 27*mm, 27*mm]

    text_location_x = 4*mm
    text_location_y = 2.85*mm

    # Shifts
    shift_x = shift_mat[0]
    shift_y = shift_mat[1]
    position_mat_x = [x + shift_x for x in position_mat_x]
    position_mat_y = [y + shift_y for y in position_mat_y]
    cross_x = [x + shift_x for x in cross_x]
    cross_y = [y + shift_y for y in cross_y]
    text_location_x = text_location_x + shift_x
    text_location_y = text_location_y + shift_y

    # Creating a pinhole array
    pat_list = rect_hole_array(position_mat_x, position_mat_y, dim)
    mask.add(pat_list)

    # Alignment Cross
    for idx in range(0, len(cross_x)):
        a_cross = alignment_cross([cross_x[idx], cross_y[idx]], [200*um, 50*um])
        mask.add(a_cross)

    # Text
    text_element = gdspy.Text('Pinhole Array ({} um)'.format(str(dim[0])), size=300, position=[text_location_x, text_location_y])
    mask.add(text_element)
    
    return None

def dithered_pat(mask, position_mat, shift_mat, dither_path, num, dither_size, translate_coords, text):
    position_mat_x = position_mat[0]
    position_mat_y = position_mat[1]

    cross_x = [3*mm, 27*mm, 3*mm, 27*mm]
    cross_y = [3*mm, 3*mm, 27*mm, 27*mm]
#    num_rep = [10,5]

    text_location_x = 4*mm
    text_location_y = 2.85*mm
    size_text = 300

    # Shifts
    shift_x = shift_mat[0]
    shift_y = shift_mat[1]
    position_mat_x = [x + shift_x for x in position_mat_x]
    position_mat_y = [y + shift_y for y in position_mat_y]
    cross_x = [x + shift_x for x in cross_x]
    cross_y = [y + shift_y for y in cross_y]
    text_location_x = text_location_x + shift_x
    text_location_y = text_location_y + shift_y

#    index = 0
    # Creating a Single pattern
    coord_path = dither_path
    pattern = dither_func(coord_path, num, position_mat_x, position_mat_y, dither_size, translate_coords)

    mask.add(pattern)

    # Alignment Cross
    for idx in range(0, len(cross_x)):
        a_cross = alignment_cross([cross_x[idx], cross_y[idx]], [200*um, 50*um])
        mask.add(a_cross)

    # Text
    text_element = gdspy.Text(text, size=size_text, position=[text_location_x, text_location_y])
    mask.add(text_element)

def position_pattern(mask, mask_positions, shift_mat, text):

    cross_x = [1*mm, 29*mm, 1*mm, 29*mm]
    cross_y = [1*mm, 1*mm, 29*mm, 29*mm]

    text_location_x = 4*mm
    text_location_y = 0.85*mm
    size_text = 300

    # Shifts
    shift_x = shift_mat[0]
    shift_y = shift_mat[1]

    # position_mat_x = [x + shift_x for x in position_mat_x]
    # position_mat_y = [y + shift_y for y in position_mat_y]
    cross_x = [x + shift_x for x in cross_x]
    cross_y = [y + shift_y for y in cross_y]
    text_location_x = text_location_x + shift_x
    text_location_y = text_location_y + shift_y

    pattern = position_func(mask_positions, shift_mat)

    mask.add(pattern)

    # Alignment Cross
    for idx in range(0, len(cross_x)):
        a_cross = alignment_cross([cross_x[idx], cross_y[idx]], [200*um, 50*um])
        mask.add(a_cross)

    # Text
    text_element = gdspy.Text(text, size=size_text, position=[text_location_x, text_location_y])
    mask.add(text_element)

def calib_mask(mask, position_mat, shift_mat, calib_path, num, dither_size, translate_coords, text):
    position_mat_x = position_mat[0]
    position_mat_y = position_mat[1]

    cross_x = [3*mm, 27*mm, 3*mm, 27*mm]
    cross_y = [3*mm, 3*mm, 27*mm, 27*mm]
#    num_rep = [10,5]

    text_location_x = 4*mm
    text_location_y = 2.85*mm
    size_text = 300

    # Shifts
    shift_x = shift_mat[0]
    shift_y = shift_mat[1]
    position_mat_x = [x + shift_x for x in position_mat_x]
    position_mat_y = [y + shift_y for y in position_mat_y]
    cross_x = [x + shift_x for x in cross_x]
    cross_y = [y + shift_y for y in cross_y]
    text_location_x = text_location_x + shift_x
    text_location_y = text_location_y + shift_y

#    index = 0
    # Creating a Single pattern
    coord_path = calib_path
    pattern = set_rect(coord_path, num, position_mat_x, position_mat_y, dither_size, translate_coords)

    mask.add(pattern)

    # Alignment Cross
    for idx in range(0, len(cross_x)):
        a_cross = alignment_cross([cross_x[idx], cross_y[idx]], [200*um, 50*um])
        mask.add(a_cross)

    # Text
    text_element = gdspy.Text(text, size=size_text, position=[text_location_x, text_location_y])
    mask.add(text_element)