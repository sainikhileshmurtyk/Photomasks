# -*- coding: utf-8 -*-
"""
Code used from:
https://github.com/tromero/BayerMatrix/blob/master/MakeBayer.py

"""
import numpy as np
import matplotlib.pyplot as plt
# %%

def InitBayer(x, y, size, value, step,
              matrix = [[]]):
    if matrix == [[]]:
        matrix = [[0 for i in range(size)]for i in range(size)]
    
    if (size == 1):
        matrix[int(y)][int(x)] = value
        return
    
    half = size/2
    
    #subdivide into quad tree and call recursively
    #pattern is TL, BR, TR, BL
    InitBayer(x,      y,      half, value+(step*0), step*4, matrix)
    InitBayer(x+half, y+half, half, value+(step*1), step*4, matrix)
    InitBayer(x+half, y,      half, value+(step*2), step*4, matrix)
    InitBayer(x,      y+half, half, value+(step*3), step*4, matrix)
    return matrix

def MakeBayer(matrixSize, savePng, pngTileCount):
    pngFilename = 'bayer%d.png' % matrixSize
    if (pngTileCount > 1):
        pngFilename = 'bayer%dtile%d.png' % (matrixSize, pngTileCount)

    matrix = InitBayer(0, 0, matrixSize, 0, 1)
    
    if (savePng):
        from PIL import Image
        brightnessMultiplier = {16:1,8:4,4:16,2:64}
        img = Image.new('RGB', (matrixSize*pngTileCount,
                                matrixSize*pngTileCount))
        imgData = img.load()
        for y in range(img.height):
            for x in range(img.width):
                value = matrix[y % matrixSize][x % matrixSize]
                value *= brightnessMultiplier[matrixSize]
                color = (value, value, value)
                imgData[x,y] = color
        img.save(pngFilename, "PNG")
        print('Saved %s' % pngFilename)
    else:
        print('Bayer Matrix %s' % matrixSize)
        print(matrix)
        

def bayer_position(spacing,value,bayer_matrix):
    (pv,ph) = np.where(bayer_matrix<value)
    pv = pv*spacing
    ph = ph*spacing
    return (pv,ph)
        
# %%
def dither_16x(values,spacing): 
    # values is a list of values to be dithered into a 16x16 sq., from 0 to 1
    # spacing is the size (in mm) of the smallest feature size 
    #
    # e.g. spacing = 0.0125 means a 16*0.0125 = 0.2 mm dithered block 
    # that is centered at a coordinate of 0,0
    # 
    # the output is three lists of numpy arrays, corresponding to the vertical 
    # position, the horizontal position, and the size of the blocks

    # 0: y, 1:x, 2: size of block
    
    bayer_matrix  = np.array(InitBayer(0, 0, 16, 0, 1))
    values = np.int32(np.round(np.array(values)*256))
    
    pos_v = []
    pos_h = []
    sz = []
    for val in values:
        if(val==256):
            pos_v.append(np.array([0]))
            pos_h.append(np.array([0]))
            sz.append(np.array([16*spacing]))
        else:
            (pv,ph) = bayer_position(spacing,val,bayer_matrix)
            pos_v.append(pv-7.5*spacing)
            pos_h.append(ph-7.5*spacing)
            sz.append(spacing*np.ones_like(pv))

    return (pos_v,pos_h,sz)
    
# %%
(pos_v,pos_h,sz) = dither_16x([0.23,0.15,1,0,0.99,1],0.0125)