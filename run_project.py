"""
    PROIECT MOZAIC
"""

# Parametrii algoritmului sunt definiti in clasa Parameters.
from parameters import *
from build_mosaic import *
import timeit
import numpy as np
import os

import cv2 as cv


dir_path = './..data/imaginiTest/'
filenames = [('./../data/imaginiTest/ferrari.jpeg', 'ferrari'),
             ('./../data/imaginiTest/adams.JPG', 'adams'),
             ('./../data/imaginiTest/liberty.jpg', 'liberty'),
             ('./../data/imaginiTest/obama.jpeg', 'obama'),
             ('./../data/imaginiTest/romania.jpeg', 'romania'),
             ('./../data/imaginiTest/tomJerry.jpeg', 'tomJerry')]
small_images_path = './../data/colectie/'
sizes = [25, 50, 75, 100]
layouts = ["caroiaj", "aleator"]
criteria = 'distantaCuloareMedie'
hexagons = [False, True]
neighbours = [False, True]

start_time = timeit.default_timer()

#(a)
for file in filenames:
    for size in sizes:
        file_name, name = file
        size = 100
        params = Parameters(file_name)
        params.small_images_dir = small_images_path
        params.image_type = 'jpg'
        params.num_pieces_horizontal = size
        params.show_small_images = False
        params.layout = 'caroiaj'
        params.hexagon = False
        params.different_neighbours = False
        params.criterion = criteria
        mosaic = build_mosaic(params)
        cv.imwrite(name + '_' + size.__str__() + '_caroiaj.png', mosaic)
        

#(b)
size = 100
for file in filenames:
    file_name, name = file
    params = Parameters(file_name)
    params.small_images_dir = small_images_path
    params.image_type = 'jpg'
    params.num_pieces_horizontal = size
    params.show_small_images = False
    params.layout = 'aleator'
    params.hexagon = False
    params.different_neighbours = False
    params.criterion = criteria
    mosaic = build_mosaic(params)
    cv.imwrite(name + '_' + size.__str__() + '_random.png', mosaic)

#(c)
size = 100
for file in filenames:
    file_name, name = file
    params = Parameters(file_name)
    params.small_images_dir = small_images_path
    params.image_type = 'jpg'
    params.num_pieces_horizontal = size
    params.show_small_images = False
    params.layout = 'caroiaj'
    params.hexagon = False
    params.different_neighbours = True
    params.criterion = criteria
    mosaic = build_mosaic(params)
    cv.imwrite(name + '_' + size.__str__() + '_caroiaj_different_neighbours.png', mosaic)


#(d)
# cifar_names = [b'airplane', b'automobile', b'bird', b'cat', b'deer',
#                b'dog', b'frog', b'horse', b'ship', b'truck']

filenames2 = [('./../data/imaginiNoi/troian.jpg', 'troian', b'horse'),
             ('./../data/imaginiNoi/pinguini.jpg', 'pinguini', b'bird'),
             ('./../data/imaginiNoi/dacia.jpg', 'dacia', b'automobile'),
             ('./../data/imaginiNoi/snoopdogg.jpg', 'snoopdogg', b'dog'),
             ('./../data/imaginiNoi/frog.jpg', 'frog', b'frog')]
cifar_dir_path = './../data/cifar-10-batches-py/'
cifar_path = './../data/cifar-10-batches-py/data_batch_1'
size = 100
for file in filenames2:
    file_name, name, cifar_name = file
    params = Parameters(file_name)
    params.small_images_dir = cifar_dir_path
    params.image_type = 'jpg'
    params.num_pieces_horizontal = size
    params.show_small_images = False
    params.layout = 'caroiaj'
    params.hexagon = False
    params.different_neighbours = False
    params.criterion = criteria
    params.cifar = True
    params.cifar_name = cifar_name
    mosaic = build_mosaic(params)
    
    # These lines are optional if you want to resize the mosaic
    # so that the image won't occupy a tone of space
    # if params.grayscale:
    #     H, W = mosaic.shape
    # else:
    #     H, W, _ = mosaic.shape
    # mosaic = cv.resize(mosaic, (H // 4, W // 4))
    cv.imwrite(name + '_' + size.__str__() + '_cifar_' + cifar_name.decode('ascii') + '.png', mosaic)



#(e)
size = 100
for file in filenames:
    file_name, name = file
    params = Parameters(file_name)
    params.small_images_dir = small_images_path
    params.image_type = 'jpg'
    params.num_pieces_horizontal = size
    params.show_small_images = False
    params.layout = 'caroiaj'
    params.hexagon = True
    params.different_neighbours = False
    params.criterion = criteria
    mosaic = build_mosaic(params)
    cv.imwrite(name + '_' + size.__str__() + '_hexagoane.png', mosaic)


#(f)
size = 100
for file in filenames:
    file_name, name = file
    params = Parameters(file_name)
    params.small_images_dir = small_images_path
    params.image_type = 'jpg'
    params.num_pieces_horizontal = size
    params.show_small_images = False
    params.layout = 'caroiaj'
    params.hexagon = True
    params.different_neighbours = True
    params.criterion = criteria
    mosaic = build_mosaic(params)
    cv.imwrite(name + '_' + size.__str__() + '_hexagoane_different_neighbours.png', mosaic)

end_time = timeit.default_timer()
print('Entire Project running time: %f s.' % (end_time - start_time))

