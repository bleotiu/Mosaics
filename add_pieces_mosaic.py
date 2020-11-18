from parameters import *
import numpy as np
import pdb
import timeit
from scipy.spatial import cKDTree
import cv2 as cv
import random


def add_pieces_grid(params: Parameters):
    start_time = timeit.default_timer()
    img_mosaic = np.zeros(params.image_resized.shape, np.uint8)
    if not params.grayscale:
        N, H, W, C = params.small_images.shape
        h, w, c = params.image_resized.shape
    else:
        N, H, W = params.small_images.shape
        h, w = params.image_resized.shape
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal

    if params.criterion == 'aleator':
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                index = np.random.randint(low=0, high=N, size=1)
                img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W, :] = params.small_images[index]
                print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))

    elif params.criterion == 'distantaCuloareMedie':
        mean_color_pieces = np.mean(params.small_images, axis=(1, 2))
        if params.different_neighbours:
            ngh = np.zeros((params.num_pieces_vertical, params.num_pieces_horizontal)) - 1
            # neighbour matrix
            num_indexes = 3 # if we have to have different neighbours
            #we will need the 3 closest pieces in case the best two
            #pieces are already adjacent to the current position(top or left)
        else:
            num_indexes = 1 # one index is sufficient if we don't care
            #about matching neighbours
        if params.grayscale:
            mean_color_pieces = [[x, x] for x in mean_color_pieces]
        tree = cKDTree(mean_color_pieces)
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                # print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))

                patch = params.image_resized[i * H: (i + 1) * H, j * W: (j + 1) * W]

                mean_patch = np.mean(patch, axis=(0, 1))
                if params.grayscale:
                    mean_patch =[mean_patch, mean_patch]
                value, index = tree.query(mean_patch, k=num_indexes)
                if num_indexes < 2:
                    img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W] = params.small_images[index]
                else:
                    for it in range(num_indexes):
                        if i > 0:
                            if ngh[i - 1, j] == index[it]:
                                continue
                            elif j > 0:
                                if ngh[i, j - 1] == index[it]:
                                    continue
                                else:
                                    idx = index[it]
                                    break
                        elif j > 0:
                            if ngh[i, j - 1] == index[it]:
                                continue
                            else:
                                idx = index[it]
                                break
                        else:
                            idx = index[it]
                            break
                    ngh[i, j] = idx
                    img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W] = params.small_images[idx]

    else:
        print('Error! unknown option %s' % params.criterion)
        exit(-1)

    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))

    return img_mosaic


def add_pieces_random(params: Parameters):
    start_time = timeit.default_timer()
    mean_color_pieces = np.mean(params.small_images, axis=(1, 2))

    if not params.grayscale:
        N, H, W, C = params.small_images.shape
        h, w, c = params.image_resized.shape
    else:
        N, H, W = params.small_images.shape
        h, w = params.image_resized.shape
    if params.grayscale:
        img_mosaic = np.zeros((h + H, w + W), np.uint8)
    else:
        img_mosaic = np.zeros((h + H, w + W, c), np.uint8)
    used = np.zeros((h + H, w + W)) - 1
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal
    positions = list(range(h * w))
    random.shuffle(positions)
    if params.grayscale:
        mean_color_pieces = [[x, x] for x in mean_color_pieces]
    tree = cKDTree(mean_color_pieces)

    for position in positions:
        i, j = position // w, position % w
        if used[i, j] < 0:
            mean_patch = params.image_resized[i: i + H, j: j + W].mean(axis=(0, 1))
            if params.grayscale:
                mean_patch = [mean_patch, mean_patch]
            value, id = tree.query(mean_patch)
            img_mosaic[i: i + H, j: j + W] = params.small_images[id]
            used[i: i + H, j: j + W] = np.full((H, W), id)


    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))

    return img_mosaic[:h, :w]


def add_pieces_hexagon(params: Parameters):
    start_time = timeit.default_timer()
    if params.grayscale:
        N, H, W = params.small_images.shape
        h, w = params.image_resized.shape
        img_mosaic = np.zeros((h + H, w + W), np.uint8)
    else:
        N, H, W, C = params.small_images.shape
        h, w, c = params.image_resized.shape
        img_mosaic = np.zeros((h + H, w + W, c), np.uint8)

    D1, D2 = 2 * params.num_pieces_vertical + 10, params.num_pieces_horizontal * 3 // 4 + 10
    ngh = np.zeros((D1, D2)) - 1
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal
    mean_color_pieces = np.mean(params.small_images, axis=(1, 2))

    if params.grayscale:
        mask = np.full((H, W), 1)
    else:
        mask = np.full((H, W, C), 1)
    for i in range(H):
        for j in range(np.abs(i - W // 3)):
            if params.grayscale:
                mask[i, j] = 0
                mask[i, W - 1 - j] = 0
            else:
                mask[i, j] = [0, 0, 0]
                mask[i, W - 1 - j] = [0, 0, 0]

    if params.grayscale:
        not_mask = np.full((H, W), 1) - mask
    else:
        not_mask = np.full((H, W, C), 1) - mask
    start_x, start_y = 0, 0
    current_x, current_y = start_x, start_y
    if params.grayscale:
        mean_color_pieces = [[x, x] for x in mean_color_pieces]
    tree = cKDTree(mean_color_pieces)
    if params.different_neighbours:
        num_indexes = 2
    else:
        num_indexes = 1
    print(img_mosaic.shape)
    i = 0
    while current_x < h:
        current_y = start_y
        j = 0
        while current_y < w:
            patch = params.image_resized[current_x: current_x + H, current_y: current_y + W]
            patch_mean = np.mean(patch, axis=(0, 1))
            if params.grayscale:
                patch_mean = [patch_mean, patch_mean]
            value, index = tree.query(patch_mean, k=num_indexes)
            if num_indexes > 1:
                if i > 1:
                    if ngh[i - 2, j] == index[0]:
                        idx = index[1]
                    else:
                        idx = index[0]
                else:
                    idx = index[0]
            else:
                idx = index
            current_image = img_mosaic[current_x: current_x + H, current_y: current_y + W]
            img_mosaic[current_x: current_x + H, current_y: current_y + W] = (mask * params.small_images[idx]) + (not_mask * current_image)
            ngh[i, j] = idx
            j += 1
            current_y += W + W // 3 - 1
        i += 2
        current_x += H

    dx, dy = [1, -1, -2, -1, 1], [0, 0, 0, 1, 1]

    if num_indexes > 1:
        num_indexes = 5

    start_x, start_y = H // 2, 2 * W // 3
    current_x, current_y = start_x, start_y
    i = 1
    while current_x < h:
        current_y = start_y
        j = 0
        while current_y < w:
            patch = params.image_resized[current_x: current_x + H, current_y: current_y + W]
            if patch.shape != mask.shape:
                patch = cv.resize(patch, (W, H))
            patch_mean = np.mean(patch, axis=(0, 1))
            if params.grayscale:
                patch_mean = [patch_mean, patch_mean]
            value, index = tree.query(patch_mean, k=num_indexes)
            if num_indexes > 1:
                for id in index:
                    ok = True
                    for it in range(num_indexes):
                        if (-1 < i + dx[it] < D1) and (-1 < j + dy[it] < D2) and (ngh[i + dx[it], j + dy[it]] == id):
                            ok = False
                    if ok:
                        idx = id
                        break
            else:
                idx = index

            current_image = img_mosaic[current_x: current_x + H, current_y: current_y + W]
            img_mosaic[current_x: current_x + H, current_y: current_y + W] = (mask * params.small_images[idx]) + (not_mask * current_image)
            ngh[i, j] = idx
            j += 1
            current_y += W + W // 3 - 1
        i += 2
        current_x += H
    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))

    return img_mosaic[H // 2: h, W // 3: w]
