import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pdb
import random

from add_pieces_mosaic import *
from parameters import *


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def cifar_to_image(data):
    cifar_H = 32
    cifar_W = 32
    num_pixels = cifar_H * cifar_W
    current_image = np.zeros((cifar_H, cifar_W, 3), np.uint8)
    for i in range(cifar_H):
        for j in range(cifar_W):
            for k in range(3):
                current_image[i][j][k] = data[i * cifar_W + j + (2 - k) * num_pixels]
    return current_image


def cifar_images(params):
    images = []
    meta = unpickle(params.small_images_dir + 'batches.meta')[b'label_names']
    label_index = {}
    n = len(meta)
    for it in range(n):
        label_index[meta[it]] = it

    data_index = random.randint(1, 5)
    unpickled = unpickle(params.small_images_dir + 'data_batch_' + str(data_index))
    data_labels = unpickled[b'labels']
    data = unpickled[b'data']
    n = len(data_labels)
    for it in range(n):
        if data_labels[it] == label_index[params.cifar_name]:
            current_image = cifar_to_image(data[it])
            if params.grayscale:
                images.append(cv.cvtColor(current_image, cv.COLOR_BGR2GRAY))
            else:
                images.append(current_image)
    return images


def images_from_dir(path, grayscale):
    filenames = os.listdir(path)
    images = []
    for image_name in filenames:
        if grayscale:
            img = cv.cvtColor(cv.imread(path + image_name), cv.COLOR_BGR2GRAY)
        else:
            img = cv.imread(path + image_name)
        images.append(img)
    return images

def load_pieces(params: Parameters):
    # citesc toate cele N piese folosite la mozaic din directorul corespunzator
    # toate cele N imagini au aceeasi dimensiune H x W x C, unde:
    # H = inaltime, W = latime, C = nr canale (C=1  gri, C=3 color)
    # functia intoarce pieseMozaic = matrice N x H x W x C in params
    # pieseMoziac[i, :, :, :] reprezinta piesa numarul i
    if params.cifar:
        images = cifar_images(params)
    else:
        images = images_from_dir(params.small_images_dir, params.grayscale)

    # citeste imaginile din director

    if params.show_small_images:
        for i in range(10):
            for j in range(10):
                plt.subplot(10, 10, i * 10 + j + 1)
                # OpenCV reads images in BGR format, matplotlib reads images in RBG format
                im = images[i * 10 + j].copy()
                # BGR to RGB, swap the channels
                if not params.grayscale:
                    im = im[:, :, [2, 1, 0]]
                plt.imshow(im)
        plt.show()

    params.small_images = np.array(images)


def compute_dimensions(params: Parameters):
    # calculeaza dimensiunile mozaicului
    # obtine si imaginea de referinta redimensionata avand aceleasi dimensiuni
    # ca mozaicul

    # completati codul
    # calculeaza automat numarul de piese pe verticala
    if params.grayscale:
        H, W = params.image.shape
    else:
        H, W, _ = params.image.shape
    if params.grayscale:
        small_h, small_w = params.small_images[0].shape
    else:
        small_h, small_w, _ = params.small_images[0].shape

    # redimensioneaza imaginea
    new_w = params.num_pieces_horizontal * small_w
    aspect_ratio = H / W # suspect
    new_h = new_w * aspect_ratio
    params.num_pieces_vertical = int(np.ceil(new_h / small_h))
    new_h = params.num_pieces_vertical * small_h
    params.image_resized = cv.resize(params.image, (new_w, new_h))


def build_mosaic(params: Parameters):
    # incarcam imaginile din care vom forma mozaicul
    load_pieces(params)
    # return None
    # calculeaza dimensiunea mozaicului
    compute_dimensions(params)

    img_mosaic = None
    if params.layout == 'caroiaj':
        if params.hexagon is True:
            img_mosaic = add_pieces_hexagon(params)
        else:
            img_mosaic = add_pieces_grid(params)
    elif params.layout == 'aleator':
        img_mosaic = add_pieces_random(params)
    else:
        print('Wrong option!')
        exit(-1)

    return img_mosaic
