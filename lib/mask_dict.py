import os
import h5py
import numpy as np
from scipy import io
import lib.GDSTKUtils as utils

os.chdir('/home/murty/Documents/Project Directory/Neuromorphic Computing/LEDNN/Hardware/Photomasks/Code')
'''
Setting some parameters
'''
# default size unit is microns, so define handy shortcuts for other units
um = 1
mm = 1e3
cm = 1e4

'''
Loading the files
'''
# Loading the Matrix
mat_path = os.path.join("Data", "Positions", "221129 Masks", "mnistspread_bleed_221025205444_cellarray_flipboth.mat")
mat = io.loadmat(mat_path)

# Reading cell data
net0 = mat['net0cell']
net1 = mat['net1cell']
net2 = mat['net2cell']

# File locations
LED_input_file = os.path.join("Data", "Positions", "positions", "LED_positions_input.h5")
# LED_input_file = 'Data/Positions/221129 Masks/LED_positions_input_test.h5'
LED_dev1_file = os.path.join("Data", "Positions", "positions", "LED_positions_device1.h5")
LED_dev2_file = os.path.join("Data", "Positions", "positions", "LED_positions_device2.h5")

PD_dev1_file = os.path.join("Data", "Positions", "positions", "PD_positions_device_1_flipboth.h5")
PD_dev2_file = os.path.join("Data", "Positions", "positions", "PD_positions_device_2_flipboth.h5")
# PD_dev2_file = 'Data/Positions/221129 Masks/PD_positions_device_2_test.h5'

PD_output_file = os.path.join("Data", "Positions", "positions", "PD_positions_output_ideal_flipboth.h5")

# Loading the h5 file variables
LED_input_h5 = h5py.File(LED_input_file, "r")
LED_dev1_h5 = h5py.File(LED_dev1_file, "r")
LED_dev2_h5 = h5py.File(LED_dev2_file, "r")

PD_output_h5 = h5py.File(PD_output_file, "r")
PD_dev1_h5 = h5py.File(PD_dev1_file, "r")
PD_dev2_h5 = h5py.File(PD_dev2_file, "r")

## Obtaining the data and transposing
# Setting empty dictionaries
LED_input_cen, LED_dev1_cen, LED_dev2_cen, PD_dev1_cen, PD_dev2_cen, PD_output_cen = {}, {}, {}, {}, {}, {}

# Input
LED_input_cen['x'] = np.array(LED_input_h5.get('LED_cen_x'))*mm
LED_input_cen['y'] = np.array(LED_input_h5.get('LED_cen_y'))*mm
LED_input_cen['x'], LED_input_cen['y'] = np.transpose(LED_input_cen['x']), np.transpose(LED_input_cen['y'])
LED_input_cen['x'], LED_input_cen['y'] = utils.convert_coords([25*mm, 25*mm], LED_input_cen['x'], LED_input_cen['y'])


# LED1
LED_dev1_cen['x'] = np.array(LED_dev1_h5.get('LED_cen_x'))*mm
LED_dev1_cen['y'] = np.array(LED_dev1_h5.get('LED_cen_y'))*mm
LED_dev1_cen['x'], LED_dev1_cen['y'] = np.transpose(LED_dev1_cen['x']), np.transpose(LED_dev1_cen['y'])
LED_dev1_cen['x'], LED_dev1_cen['y'] = utils.convert_coords([25*mm, 25*mm], LED_dev1_cen['x'], LED_dev1_cen['y'])

# LED2
LED_dev2_cen['x'] = np.array(LED_dev2_h5.get('LED_cen_x'))*mm
LED_dev2_cen['y'] = np.array(LED_dev2_h5.get('LED_cen_y'))*mm
LED_dev2_cen['x'], LED_dev2_cen['y'] = np.transpose(LED_dev2_cen['x']), np.transpose(LED_dev2_cen['y'])
LED_dev2_cen['x'], LED_dev2_cen['y'] = utils.convert_coords([25*mm, 25*mm], LED_dev2_cen['x'], LED_dev2_cen['y'])

# PD1 
PD_dev1_cen['x'] = np.array(PD_dev1_h5.get('PD_cen_x'))*mm
PD_dev1_cen['y'] = np.array(PD_dev1_h5.get('PD_cen_y'))*mm
PD_dev1_cen['x'], PD_dev1_cen['y'] = np.transpose(PD_dev1_cen['x']), np.transpose(PD_dev1_cen['y'])
PD_dev1_cen['x'], PD_dev1_cen['y'] = utils.convert_coords([25*mm, 25*mm], PD_dev1_cen['x'], PD_dev1_cen['y'])


# PD2
PD_dev2_cen['x'] = np.array(PD_dev2_h5.get('PD_cen_x'))*mm
PD_dev2_cen['y'] = np.array(PD_dev2_h5.get('PD_cen_y'))*mm
PD_dev2_cen['x'], PD_dev2_cen['y'] = np.transpose(PD_dev2_cen['x']), np.transpose(PD_dev2_cen['y'])
PD_dev2_cen['x'], PD_dev2_cen['y'] = utils.convert_coords([25*mm, 25*mm], PD_dev2_cen['x'], PD_dev2_cen['y'])

# Output
PD_output_cen['x'] = np.array(PD_output_h5.get('PD_cen_x'))*mm
PD_output_cen['y'] = np.array(PD_output_h5.get('PD_cen_y'))*mm
PD_output_cen['x'], PD_output_cen['y'] = np.transpose(PD_output_cen['x']), np.transpose(PD_output_cen['y'])
PD_output_cen['x'], PD_output_cen['y'] = utils.convert_coords([25*mm, 25*mm], PD_output_cen['x'], PD_output_cen['y'])


'''
Creating the dictionary objects
'''


# Pinhole Mask Parameters
ph_mask_01 = {
    "type": "ph_array",
    "led_center": LED_input_cen,
    "dims": (150*um, 150*um),
    "title": "Pinhole Array 0-1"
}

ph_mask_shifted_01 = {
    "type": "ph_array",
    "led_center": LED_input_cen,
    "dims": (150*um, 150*um),
    "magnification": 10,
    "title": "Shifted Pinhole Array 0-1"
}

ph_mask_shifted_12 = {
    "type": "ph_array",
    "led_center": LED_dev2_cen,
    "dims": (150*um, 150*um),
    "magnification": 12.5,
    "title": "Shifted Pinhole Array 1-2"
}


# Chequerboard Mask Parameters
cb_mask_01 = {
    "type": "cb_array",
    "magnification": 10,
    "led_center": LED_input_cen,
    "pd_center": PD_dev2_cen,
    "box_size": (150*um, 150*um),
    "title": 'Chequerboard Mask (150um) 0-1'
}

cb_mask_12 = {
    "type": "cb_array",
    "magnification": 12.5,
    "led_center": LED_dev2_cen,
    "pd_center": PD_dev1_cen,
    "box_size": (150*um, 150*um),
    "title": 'Chequerboard Mask (150um) 1-2'
}

cb_mask_23 = {
    "type": "cb_array",
    "magnification": 12.5,
    "led_center": LED_dev1_cen,
    "pd_center": PD_output_cen,
    "box_size": (150*um, 150*um),
    "title": 'Chequerboard Mask (150um) 2-3'
}


# Weight Mask Parameters
weight_mask_184_01 = {
    "type": "weight_mask",
    "net": net0,
    "led_center": LED_input_cen,
    "pd_center": PD_dev2_cen,
    "dither_size": 184*um,
    "magnification": 10,
    "title" : 'Weights (184um) 0-1'
}

weight_mask_184_12 = {
    "type": "weight_mask",
    "net": net1,
    "led_center": LED_dev2_cen,
    "pd_center": PD_dev1_cen,
    "dither_size": 184*um,
    "magnification": 12.5,
    "title" : 'Weights (184um) 1-2'
}

weight_mask_184_23 = {
    "type": "weight_mask",
    "net": net2,
    "led_center": LED_dev1_cen,
    "pd_center": PD_output_cen,
    "dither_size": 184*um,
    "magnification": 12.5,
    "title" : 'Weights (184um) 2-3'
}


weight_mask_200_01 = {
    "type": "weight_mask",
    "net": net0,
    "led_center": LED_input_cen,
    "pd_center": PD_dev2_cen,
    "dither_size": 200*um,
    "magnification": 10,
    "title" : 'Weights (200um) 0-1'
}

weight_mask_200_12 = {
    "type": "weight_mask",
    "net": net1,
    "led_center": LED_dev2_cen,
    "pd_center": PD_dev1_cen,
    "dither_size": 200*um,
    "magnification": 12.5,
    "title" : 'Weights (200um) 1-2' 
}

weight_mask_200_23 = {
    "type": "weight_mask",
    "net": net2,
    "led_center": LED_dev1_cen,
    "pd_center": PD_output_cen,
    "dither_size": 200*um,
    "magnification": 12.5,
    "title" : 'Weights (200um) 2-3'
}


weight_mask_216_01 = {
    "type": "weight_mask",
    "net": net0,
    "led_center": LED_input_cen,
    "pd_center": PD_dev2_cen,
    "dither_size": 216*um,
    "magnification": 10,
    "title" : 'Weights (216um) 0-1'
}

weight_mask_216_12 = {
    "type": "weight_mask",
    "net": net1,
    "led_center": LED_dev2_cen,
    "pd_center": PD_dev1_cen,
    "dither_size": 216*um,
    "magnification": 12.5,
    "title" : 'Weights (216um) 1-2'
}

weight_mask_216_23 = {
    "type": "weight_mask",
    "net": net2,
    "led_center": LED_dev1_cen,
    "pd_center": PD_output_cen,
    "dither_size": 216*um,
    "magnification": 12.5,
    "title" : 'Weights (216um) 2-3'
}