###############################################################################
# MIT License
#
# Copyright (c) 2020 Daniel Vitor Ruiz
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################

import os
import cv2
import numpy as np
from PIL import Image
import time
import random

# for fancy parameterization
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(
        description='Compute resulting image using IDA techinique for the DUTS dataset')

    parser.add_argument(
        '-obj_path', '--obj_path',
        type=str, default="/home/bakrinski/workspace/datasets/duts/training/images/",
        help='OBJ_FOLDER_IMG input images path'
    )

    parser.add_argument(
        '-obj_mask_path', '--obj_mask_path',
        type=str, default="/home/bakrinski/workspace/datasets/duts/training/colored_masks/",
        help='OBJ_FOLDER_MASK input masks path'
    )

    parser.add_argument(
        '-bg_path', '--bg_path',
        type=str, default="/home/bakrinski/workspace/datasets/duts/training/inpaintedv2_nogrid/",
        help='BG_FOLDER_IMG background images path'
    )

    parser.add_argument(
        '-index_obj_path', '--index_obj_path',
        type=str, default="dataset.txt",
        help='LIST_OF_N_OBJECTS filepath for the file containing per line a indice, e.g. "dataset.txt" resulting from computeKnn.py'
    )

    parser.add_argument(
        '-index_bg_path', '--index_bg_path',
        type=str, default="indices_cosine.txt",
        help='LIST_OF_INDICES filepath for the file containing per line a indice, e.g. "indices_cosine.txt" resulting from computeKnn.py'
    )

    parser.add_argument(
        '-out_path', '--out_path',
        type=str, default="output/",
        help='output path containing a folder named images and masks, e.g."output/" '
    )

    parser.add_argument(
        '-seed', '--seed',
        type=int, default=22,
        help='seed number for the pseudo-random computation'
    )

    parser.add_argument(
        '-size', '--size',
        type=int, default=10553,
        help='number of images in the dataset'
    )

    parser.add_argument(
        '-n_bgs', '--n_bgs',
        type=int, default=1,
        help='N_OF_BACKGROUNDS'
    )

    parser.add_argument(
        '-n_ops', '--n_ops',
        type=int, default=1,
        help='N_OF_OPS'
    )

    parser.add_argument(
        '-dataset_file', '--dataset_file',
        type=str, default="/home/bakrinski/workspace/datasets/duts/training/paths.txt",
        help='DATASET_FILE'
    )

    return parser.parse_args()

# SETTINGS


# CALL PARSER
args = parse_args()
#

OBJ_FOLDER_IMG = args.obj_path
OBJ_FOLDER_MASK = args.obj_mask_path
BG_FOLDER_IMG = args.bg_path
OUTPUT_FOLDER_IMG = args.out_path + "images/"
OUTPUT_FOLDER_MASK = args.out_path + "masks/"
LIST_OF_N_OBJECTS = args.index_obj_path
N_OBJECT = args.size
N_OF_BACKGROUNDS = args.n_bgs
N_OF_OPS = args.n_ops
LIST_OF_INDICES = args.index_bg_path
DATASET_FILE = args.dataset_file

kernelErode = np.ones((3, 3), np.uint8)

maxH = 512
maxW = 512

random.seed(args.seed)
np.random.seed(args.seed)
# noise_scale = np.random.uniform(low=0.975, high=1.025, size=N_OBJECT)
# noise_x = np.random.uniform(low=0.0, high=1.0, size=N_OBJECT)
# noise_y = np.random.uniform(low=0.0, high=1.0, size=N_OBJECT)
#


def computeMaximalDistanceForTranslation(obj, obj_mask, bg, newYmax, newYmin, newXmax, newXmin, newOrigin, border, M):
    import torch
    from scipy.spatial import distance
    import skimage.feature as ft

    DEVICE = "cpu"
    NBINS = 64

    def getHistogramsTorch(img):

        img = Image.fromarray((img).astype(np.uint8))

        h, s, v = img.convert('HSV').split()
        h = np.array(h)
        s = np.array(s)
        v = np.array(v)

        histH = torch.histc(torch.from_numpy(h).float().to(
            DEVICE), bins=NBINS, min=0.0, max=255.0)
        histS = torch.histc(torch.from_numpy(s).float().to(
            DEVICE), bins=NBINS, min=0.0, max=255.0)
        histV = torch.histc(torch.from_numpy(v).float().to(
            DEVICE), bins=NBINS, min=0.0, max=255.0)

        imgGray = img.convert('L')

        # # settings for LBP
        # # radius = 3
        # # n_points = 8 * radius
        # # METHOD = 'uniform'
        # # lbp = ft.local_binary_pattern(imgGray, n_points, radius, METHOD)
        lbp = ft.local_binary_pattern(imgGray, 24, 3, 'uniform')

        histLBP = torch.histc(torch.from_numpy(lbp).float().to(
            DEVICE), bins=NBINS, min=0.0, max=255.0)

        hist = torch.stack((histH, histS, histV, histLBP))
        hist = hist.view(-1)

        # img.save("tmp/bgTmp"+str((time.time()))+".png")

        return hist.numpy()

    def getHistogramsWithMaskTorch(img, mask):

        img = Image.fromarray((img).astype(np.uint8))
        mask = Image.fromarray((mask).astype(np.uint8))

        if img.mode == "L":
            img = img.convert('RGB')

        if mask.mode != "L":
            mask = mask.convert('L')

        h, s, v = img.convert('HSV').split()
        h = np.array(h) * mask
        s = np.array(s) * mask
        v = np.array(v) * mask

        histH = torch.histc(torch.from_numpy(h).float().to(
            DEVICE), bins=NBINS, min=0.0, max=255.0)
        histS = torch.histc(torch.from_numpy(s).float().to(
            DEVICE), bins=NBINS, min=0.0, max=255.0)
        histV = torch.histc(torch.from_numpy(v).float().to(
            DEVICE), bins=NBINS, min=0.0, max=255.0)

        imgGray = img.convert('L')

        # # settings for LBP
        # # radius = 3
        # # n_points = 8 * radius
        # # METHOD = 'uniform'
        # # lbp = ft.local_binary_pattern(imgGray, n_points, radius, METHOD)
        lbp = ft.local_binary_pattern(imgGray, 24, 3, 'uniform')

        histLBP = torch.histc(torch.from_numpy(lbp).float().to(
            DEVICE), bins=NBINS, min=0.0, max=255.0)

        hist = torch.stack((histH, histS, histV, histLBP))
        hist = hist.view(-1)

        # imgGray.save("tmp/graytmp"+str(int(time.time()))+".png")
        # mask.save("tmp/masktmp"+str(int(time.time()))+".png")

        return hist.numpy()

    # print("-------------")
    fd = getHistogramsWithMaskTorch(obj, obj_mask)

    # imgtmp = Image.fromarray((bg).astype(np.uint8))
    # imgtmp.save("tmp/bgFullTmp"+str((time.time()))+".png")

    bg_h, bg_w, _ = bg.shape
    obj_h, obj_w, _ = obj.shape
    # print("bg.shape=", bg.shape)
    # print("obj.shape=", obj.shape)

    h = int(np.ceil(bg_h / obj_h))
    w = int(np.ceil(bg_w / obj_w))
    # print("h=", h)
    # print("w=", w)

    cell = np.zeros((h, w, NBINS * 4))
    # print(cell.shape)

    disMax = 0.0
    startXmax = 0.0
    endXmax = 0.0
    startYmax = 0.0
    endYmax = 0.0

    for i in range(0, h):
        for j in range(0, w):

            starty = min((i) * obj_h, bg_h - obj_h)
            endy = min((i + 1) * obj_h, bg_h)

            startx = min((j) * obj_w, bg_w - obj_w)
            endx = min((j + 1) * obj_w, bg_w)

            if((endx - startx) < obj_w):
                startx = bg_w - obj_w
                endx = bg_w
            if((endy - starty) < obj_h):
                starty = bg_h - obj_h
                endy = bg_h

            # print("starty=",starty,"endy=",endy,"startx=",startx,"endx=",endx)

            # print(getHistogramsTorch(bg[starty:endy,startx:endx]))
            cell[i][j] = getHistogramsTorch(bg[starty:endy, startx:endx])
            dis = distance.cosine(cell[i][j], fd)
            if(dis > disMax):
                disMax = dis
                startXmax = startx
                endXmax = endx
                startYmax = starty
                endYmax = endy
            # print(dis)

    # print("Max=", disMax, "imax=", iMax, "jmax=", jMax)
    # print("-------------")

    # Begin translate LOGIC
    # newOrigin[0]#y
    # newOrigin[1]#x

    posY = startYmax + newOrigin[0]
    posX = startXmax + newOrigin[1]

    # y
    M[1][2] += posY - newYmin
    # x
    M[0][2] += posX - newXmin

    # End translate LOGIC

    return M


def randomTranslateInside(newYmax, newYmin, newXmax, newXmin, newOrigin, border, M):
    noise_x = np.random.uniform(low=0.0, high=1.0)
    noise_y = np.random.uniform(low=0.0, high=1.0)
    # check if bbox can move in y
    if((newYmax - newYmin) < border[0]):
        # check the direction of free space
        if((newYmax) < newOrigin[0] + border[0]):
            if((newYmin) > newOrigin[0]):
                freeSpacePos = (newOrigin[0] + border[0]) - newYmax
                freeSpaceNeg = newYmin - newOrigin[0]

                luck = np.random.randint(low=0, high=2)
                if(luck == 0):
                    M[1][2] += np.floor(noise_y * freeSpacePos)
                else:
                    M[1][2] -= np.floor(noise_y * freeSpaceNeg)

            else:
                freeSpace = (newOrigin[0] + border[0]) - newYmax
                M[1][2] += np.floor(noise_y * freeSpace)
        else:
            if((newYmin) > newOrigin[0]):
                freeSpace = newYmin - newOrigin[0]
                M[1][2] -= np.floor(noise_y * freeSpace)

    if((newXmax - newXmin) < border[1]):
        # check the direction of free space
        if((newXmax) < newOrigin[1] + border[1]):
            if((newXmin) > newOrigin[1]):
                freeSpacePos = (newOrigin[1] + border[1]) - newXmax
                freeSpaceNeg = newXmin - newOrigin[1]

                luck = np.random.randint(low=0, high=2)
                if(luck == 0):
                    M[0][2] += np.floor(noise_x * freeSpacePos)
                else:
                    M[0][2] -= np.floor(noise_x * freeSpaceNeg)

            else:
                freeSpace = (newOrigin[1] + border[1]) - newXmax
                M[0][2] += np.floor(noise_x * freeSpace)
        else:
            if((newXmin) > newOrigin[1]):
                freeSpace = newXmin - newOrigin[1]
                M[0][2] -= np.floor(noise_x * freeSpace)
    return M


def geometricOp2(resizedImg, resizedMask, bgOriginalshape, op, globalIndex, bgOriginalImg):
    #######################################################
    diffH = int((resizedImg.shape[0] - bgOriginalshape[0]) / 2)
    diffW = int((resizedImg.shape[1] - bgOriginalshape[1]) / 2)
    ####
    # compute bounding box
    ymin, ymax, xmin, xmax = bbox(resizedMask)

    # compute obj size
    propX = (xmax - xmin)
    propY = (ymax - ymin)

    # compute area
    areaOBJ = propX * propY
    areaIMG = bgOriginalshape[0] * bgOriginalshape[1]

    # compute obj size in relation to bg
    prop = areaOBJ / areaIMG
    ###

    # scale definition
    op = globalIndex % 3

    if(op == 0):
        # beta = 0.05 * noise_scale[globalIndex]
        beta = np.random.uniform(low=0.075, high=0.10)
    if(op == 1):
        # beta = 0.15 * noise_scale[globalIndex]
        beta = np.random.uniform(low=0.10, high=0.20)
    if(op == 2):
        # beta = 0.25 * noise_scale[globalIndex]
        beta = np.random.uniform(low=0.20, high=0.30)
    # if(op == 3):
    #     beta = 0.35 * noise_scale[globalIndex]

    scale = np.sqrt((beta * areaIMG) / areaOBJ)
    ###

    diffx = ((xmax - xmin) / 2)
    diffy = ((ymax - ymin) / 2)
    centerx = xmin + diffx
    centery = ymin + diffy

    pts1 = np.float32([[xmin, ymin], [xmax, ymin], [xmin, ymax]])

    newXmin = centerx - diffx * scale
    newXmax = centerx + diffx * scale

    newYmin = centery - diffy * scale
    newYmax = centery + diffy * scale

    # LOGIC HERE
    newOrigin = [diffH, diffW]
    border = [bgOriginalshape[0], bgOriginalshape[1]]

    # check if the aspect of the object is the same as the bg
    obj_orientation = -1
    bg_orientation = -1

    if(diffx >= diffy):
        obj_orientation = 0
    else:
        obj_orientation = 1

    if(bgOriginalshape[1] >= bgOriginalshape[0]):
        bg_orientation = 0
    else:
        bg_orientation = 1

    # check if can fit
    if((newYmax - newYmin <= border[0])and(newXmax - newXmin <= border[1])):
        # ok then it can fit
        # but does it need translation?

        pts2 = np.float32(
            [[newXmin, newYmin], [newXmax, newYmin], [newXmin, newYmax]])

        M = cv2.getAffineTransform(pts1, pts2)

        # origin of object must be >= newOrigin
        if(newYmin <= newOrigin[0]):
            local_diff_y = newOrigin[0] - newYmin
            M[1][2] += (local_diff_y)

        if(newXmin <= newOrigin[1]):
            local_diff_x = newOrigin[1] - newXmin
            M[0][2] += (local_diff_x)

        # maxdim must be <= border with the correct origin
        if(newYmax >= (border[0] + newOrigin[0])):
            local_diff_y = newYmax - (border[0] + newOrigin[0])
            M[1][2] -= (local_diff_y)

        if(newXmax >= (border[1] + newOrigin[1])):
            local_diff_x = newXmax - (border[1] + newOrigin[1])
            M[0][2] -= (local_diff_x)

        newXmin = xmin * M[0][0] + ymin * M[0][1] + M[0][2]
        newXmax = xmax * M[0][0] + ymax * M[0][1] + M[0][2]

        newYmin = xmin * M[1][0] + ymin * M[1][1] + M[1][2]
        newYmax = xmax * M[1][0] + ymax * M[1][1] + M[1][2]

        newXminTmp = min(newXmin, newXmax)
        newXmaxTmp = max(newXmin, newXmax)

        newYminTmp = min(newYmin, newYmax)
        newYmaxTmp = max(newYmin, newYmax)

        newXmin = newXminTmp
        newXmax = newXmaxTmp

        newYmin = newYminTmp
        newYmax = newYmaxTmp

        # NEW
        # resizedImgTmp = cv2.warpAffine(
        #     resizedImg, M, (bgOriginalshape[1], bgOriginalshape[0]), flags=cv2.INTER_LINEAR)
        # resizedMaskTmp = cv2.warpAffine(
        #     resizedMask, M, (bgOriginalshape[1], bgOriginalshape[0]), flags=cv2.INTER_NEAREST)
        resizedImgTmp = cv2.warpAffine(
            resizedImg, M, (maxW, maxH), flags=cv2.INTER_LINEAR)
        resizedMaskTmp = cv2.warpAffine(
            resizedMask, M, (maxW, maxH), flags=cv2.INTER_NEAREST)
        # print([int(newYmin),int(newYmax),int(newXmin),int(newXmax)])
        M = computeMaximalDistanceForTranslation(resizedImgTmp[int(newYmin):int(newYmax), int(newXmin):int(newXmax)], resizedMaskTmp[int(
            newYmin):int(newYmax), int(newXmin):int(newXmax)], bgOriginalImg, newYmax, newYmin, newXmax, newXmin, newOrigin, border, M)
        #######

        # M = randomTranslateInside(
        #     newYmax, newYmin, newXmax, newXmin, newOrigin, border, M)

        resizedImg = cv2.warpAffine(
            resizedImg, M, (maxW, maxH), flags=cv2.INTER_LINEAR)
        resizedMask = cv2.warpAffine(
            resizedMask, M, (maxW, maxH), flags=cv2.INTER_NEAREST)
        # resizedImg = cv2.warpAffine(
        #     resizedImg, M, (bgOriginalshape[1], bgOriginalshape[0]), flags=cv2.INTER_LINEAR)
        # resizedMask = cv2.warpAffine(
        #     resizedMask, M, (bgOriginalshape[1], bgOriginalshape[0]), flags=cv2.INTER_NEAREST)
    else:
        # it cannot fit
        # resize
        if(obj_orientation == bg_orientation):
            print("same orientatation")
            print("scale=",np.sqrt((beta * areaIMG) / areaOBJ))

            # scale must consider translation
            scale = min((border[0]) / (ymax - ymin),
                        (border[1]) / (xmax - xmin))
            scale = scale*0.5
            #
            newXmin = centerx - diffx * scale
            newXmax = centerx + diffx * scale

            newYmin = centery - diffy * scale
            newYmax = centery + diffy * scale

            pts2 = np.float32(
                [[newXmin, newYmin], [newXmax, newYmin], [newXmin, newYmax]])

            M = cv2.getAffineTransform(pts1, pts2)

            # origin of object must be >= newOrigin
            if(newYmin <= newOrigin[0]):
                local_diff_y = newOrigin[0] - newYmin
                M[1][2] += (local_diff_y)

            if(newXmin <= newOrigin[1]):
                local_diff_x = newOrigin[1] - newXmin
                M[0][2] += (local_diff_x)

            # maxdim must be <= border with the correct origin
            if(newYmax >= (border[0] + newOrigin[0])):
                local_diff_y = newYmax - (border[0] + newOrigin[0])
                M[1][2] -= (local_diff_y)

            if(newXmax >= (border[1] + newOrigin[1])):
                local_diff_x = newXmax - (border[1] + newOrigin[1])
                M[0][2] -= (local_diff_x)

            newXmin = xmin * M[0][0] + ymin * M[0][1] + M[0][2]
            newXmax = xmax * M[0][0] + ymax * M[0][1] + M[0][2]

            newYmin = xmin * M[1][0] + ymin * M[1][1] + M[1][2]
            newYmax = xmax * M[1][0] + ymax * M[1][1] + M[1][2]

            newXminTmp = min(newXmin, newXmax)
            newXmaxTmp = max(newXmin, newXmax)

            newYminTmp = min(newYmin, newYmax)
            newYmaxTmp = max(newYmin, newYmax)

            newXmin = newXminTmp
            newXmax = newXmaxTmp

            newYmin = newYminTmp
            newYmax = newYmaxTmp

            # NEW
            # resizedImgTmp = cv2.warpAffine(
            #     resizedImg, M, (bgOriginalshape[1], bgOriginalshape[0]), flags=cv2.INTER_LINEAR)
            # resizedMaskTmp = cv2.warpAffine(
            #     resizedMask, M, (bgOriginalshape[1], bgOriginalshape[0]), flags=cv2.INTER_NEAREST)
            resizedImgTmp = cv2.warpAffine(
                resizedImg, M, (maxW, maxH), flags=cv2.INTER_LINEAR)
            resizedMaskTmp = cv2.warpAffine(
                resizedMask, M, (maxW, maxH), flags=cv2.INTER_NEAREST)
            # print([int(newYmin),int(newYmax),int(newXmin),int(newXmax)])
            M = computeMaximalDistanceForTranslation(resizedImgTmp[int(newYmin):int(newYmax), int(newXmin):int(newXmax)], resizedMaskTmp[int(
                newYmin):int(newYmax), int(newXmin):int(newXmax)], bgOriginalImg, newYmax, newYmin, newXmax, newXmin, newOrigin, border, M)
            #######

            # M = randomTranslateInside(
            #     newYmax, newYmin, newXmax, newXmin, newOrigin, border, M)

            resizedImg = cv2.warpAffine(
                resizedImg, M, (maxW, maxH), flags=cv2.INTER_LINEAR)
            resizedMask = cv2.warpAffine(
                resizedMask, M, (maxW, maxH), flags=cv2.INTER_NEAREST)

        else:
            print("different orientatation")
            # print("scale=",np.sqrt((beta * areaIMG) / areaOBJ))
            # check if a rotated obj fits

            idxmod = np.random.randint(low=0, high=2)
            if(idxmod == 0):
                degrees = -90
            if(idxmod == 1):
                degrees = 90

            M = cv2.getRotationMatrix2D(((maxW / 2), (maxH / 2)), degrees, 1)

            newXmin = xmin * M[0][0] + ymin * M[0][1] + M[0][2]
            newXmax = xmax * M[0][0] + ymax * M[0][1] + M[0][2]

            newYmin = xmin * M[1][0] + ymin * M[1][1] + M[1][2]
            newYmax = xmax * M[1][0] + ymax * M[1][1] + M[1][2]

            newXminTmp = min(newXmin, newXmax)
            newXmaxTmp = max(newXmin, newXmax)

            newYminTmp = min(newYmin, newYmax)
            newYmaxTmp = max(newYmin, newYmax)

            newXmin = newXminTmp
            newXmax = newXmaxTmp

            newYmin = newYminTmp
            newYmax = newYmaxTmp

            # scale must consider translation
            scale = min((border[0]) / (newYmax - newYmin),
                        (border[1]) / (newXmax - newXmin))
            scale=scale*0.5
            #

            M[0][0] *= scale
            M[0][1] *= scale
            M[1][0] *= scale
            M[1][1] *= scale

            newXmin = xmin * M[0][0] + ymin * M[0][1] + M[0][2]
            newXmax = xmax * M[0][0] + ymax * M[0][1] + M[0][2]

            newYmin = xmin * M[1][0] + ymin * M[1][1] + M[1][2]
            newYmax = xmax * M[1][0] + ymax * M[1][1] + M[1][2]

            newXminTmp = min(newXmin, newXmax)
            newXmaxTmp = max(newXmin, newXmax)

            newYminTmp = min(newYmin, newYmax)
            newYmaxTmp = max(newYmin, newYmax)

            newXmin = newXminTmp
            newXmax = newXmaxTmp

            newYmin = newYminTmp
            newYmax = newYmaxTmp

            # origin of object must be >= newOrigin
            if(newYmin <= newOrigin[0]):
                local_diff_y = newOrigin[0] - newYmin
                M[1][2] += (local_diff_y)

            if(newXmin <= newOrigin[1]):
                local_diff_x = newOrigin[1] - newXmin
                M[0][2] += (local_diff_x)

            # maxdim must be <= border with the correct origin
            if(newYmax >= (border[0] + newOrigin[0])):
                local_diff_y = newYmax - (border[0] + newOrigin[0])
                M[1][2] -= (local_diff_y)

            if(newXmax >= (border[1] + newOrigin[1])):
                local_diff_x = newXmax - (border[1] + newOrigin[1])
                M[0][2] -= (local_diff_x)

            newXmin = xmin * M[0][0] + ymin * M[0][1] + M[0][2]
            newXmax = xmax * M[0][0] + ymax * M[0][1] + M[0][2]

            newYmin = xmin * M[1][0] + ymin * M[1][1] + M[1][2]
            newYmax = xmax * M[1][0] + ymax * M[1][1] + M[1][2]

            newXminTmp = min(newXmin, newXmax)
            newXmaxTmp = max(newXmin, newXmax)

            newYminTmp = min(newYmin, newYmax)
            newYmaxTmp = max(newYmin, newYmax)

            newXmin = newXminTmp
            newXmax = newXmaxTmp

            newYmin = newYminTmp
            newYmax = newYmaxTmp

            # NEW
            # resizedImgTmp = cv2.warpAffine(
            #     resizedImg, M, (bgOriginalshape[1], bgOriginalshape[0]), flags=cv2.INTER_LINEAR)
            # resizedMaskTmp = cv2.warpAffine(
            #     resizedMask, M, (bgOriginalshape[1], bgOriginalshape[0]), flags=cv2.INTER_NEAREST)
            resizedImgTmp = cv2.warpAffine(
                resizedImg, M, (maxW, maxH), flags=cv2.INTER_LINEAR)
            resizedMaskTmp = cv2.warpAffine(
                resizedMask, M, (maxW, maxH), flags=cv2.INTER_NEAREST)
            # print([int(newYmin),int(newYmax),int(newXmin),int(newXmax)])
            M = computeMaximalDistanceForTranslation(resizedImgTmp[int(newYmin):int(newYmax), int(newXmin):int(newXmax)], resizedMaskTmp[int(
                newYmin):int(newYmax), int(newXmin):int(newXmax)], bgOriginalImg, newYmax, newYmin, newXmax, newXmin, newOrigin, border, M)
            #######

            # M = randomTranslateInside(
            #     newYmax, newYmin, newXmax, newXmin, newOrigin, border, M)

            resizedImg = cv2.warpAffine(
                resizedImg, M, (maxW, maxH), flags=cv2.INTER_LINEAR)
            resizedMask = cv2.warpAffine(
                resizedMask, M, (maxW, maxH), flags=cv2.INTER_NEAREST)

        # print("ABORT! it cannot fit")
        # exit(1)
    ####
    # cv2.rectangle(resizedMask, (int(newXmin), int(newYmin)),
    #               (int(newXmax), int(newYmax)), (255, 255, 255), 1)
    #######################################################
    return resizedImg, resizedMask


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def resize_with_pad(image, height, width):

    def get_padding_size(image, height, width):
        # h, w, _ = image.shape
        h = image.shape[0]
        w = image.shape[1]
        # print("h=",h,"w=",w)
        top, bottom, left, right = (0, 0, 0, 0)
        if h < height:
            dh = height - h
            top = dh // 2
            bottom = dh - top
        if w < width:
            dw = width - w
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(image, height, width)
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    return constant


def resizeToOrgImg(bgOriginalshape, new):
    if(bgOriginalshape[0] < new.shape[0]):
        diffH = int((new.shape[0] - bgOriginalshape[0]) / 2)
        new = new[diffH:bgOriginalshape[0] + diffH, :, :]

    if(bgOriginalshape[1] < new.shape[1]):
        diffW = int((new.shape[1] - bgOriginalshape[1]) / 2)
        new = new[:, diffW:bgOriginalshape[1] + diffW, :]

    return new


def resizeToOrg(bgOriginalshape, new, newMask):
    if(bgOriginalshape[0] < new.shape[0]):
        diffH = int((new.shape[0] - bgOriginalshape[0]) / 2)
        new = new[diffH:bgOriginalshape[0] + diffH, :, :]
        newMask = newMask[diffH:bgOriginalshape[0] + diffH, :, :]

    if(bgOriginalshape[1] < new.shape[1]):
        diffW = int((new.shape[1] - bgOriginalshape[1]) / 2)
        new = new[:, diffW:bgOriginalshape[1] + diffW, :]
        newMask = newMask[:, diffW:bgOriginalshape[1] + diffW, :]

    return new, newMask


def loadResizedBG(index, datasetList):
    # bgName = "MSRA10K_image_{:06d}.png".format(index)
    bgName = datasetList[index][0]
    #special case
    bgName= bgName.replace(".jpg","_inpaint.jpg")

    bgFile = Image.open(BG_FOLDER_IMG + bgName)

    if bgFile.mode != "RGB":
        bgFile = bgFile.convert('RGB')

    bg = np.array(bgFile)
    bgOriginalshape = bg.shape
    resizedBg = resize_with_pad(bg, height=maxH, width=maxW)
    # resizedBg = bg
    bgFile.close()
    return resizedBg, bgOriginalshape


def load_list(file):
    mylist = []
    with open(file) as f:
        line = f.readline()
        while True:
            if not line:
                break
            line = line.split(" ")
            line[1] = line[1].replace("\n", "")

            tmp = line[0].split("/")
            line[0] = tmp[len(tmp) - 1]

            tmp = line[1].split("/")
            line[1] = tmp[len(tmp) - 1]

            mylist.append([line[0], line[1]])
            line = f.readline()
    return mylist


def main(op, multipleBgs, outPath):

    # read LIST_OF_N_OBJECTS
    arrOBJ = np.zeros(N_OBJECT, np.int)
    f = open(LIST_OF_N_OBJECTS, "r")
    for i in range(0, N_OBJECT):
        line = f.readline()
        args = line.split(" ")
        arrOBJ[i] = int(args[0])
    f.close()
    ###

    # read LIST_OF_N_OBJECTS
    arrBG = np.zeros((N_OBJECT, N_OF_BACKGROUNDS), np.int)
    f = open(LIST_OF_INDICES, "r")
    for i in range(0, N_OBJECT):
        line = f.readline()
        if line == '\n':
            arrOBJ[i] = -1
        else:
            args = line.split(" ")
            for bgindex in range(0, N_OF_BACKGROUNDS):
                arrBG[i][bgindex] = int(args[bgindex])
    f.close()
    ###

    # read DATASET_FILE
    datasetFile = DATASET_FILE
    datasetList = load_list(datasetFile)
    ###

    realI = 0

    for i in range(0, N_OBJECT, 1):
        if(arrOBJ[i] != -1):
            # imgName = "MSRA10K_image_{:06d}.jpg".format(arrOBJ[i])
            imgName = datasetList[i][0]
            imFile = Image.open(OBJ_FOLDER_IMG + imgName)

            if imFile.mode != "RGB":
                imFile = imFile.convert('RGB')

            img = np.array(imFile)

            maskName = datasetList[i][1]
            # maskName = imgName.replace(".jpg", ".png")
            # maskName = maskName.replace("image", "mask")

            maskFile = Image.open(OBJ_FOLDER_MASK + maskName)

            if maskFile.mode != "L":
                maskFile = maskFile.convert('L')

            mask = np.array(maskFile)

            mask = np.tile(mask[:, :, None], [1, 1, 3])

            resizedImg = resize_with_pad(img, height=maxH, width=maxW)
            resizedMask = resize_with_pad(mask, height=maxH, width=maxW)

            # resizedImg = img
            # resizedMask = mask

            imFile.close()
            maskFile.close()
            # print(stamp)

            resizedImgArr = [[None] * (N_OF_OPS)] * N_OF_BACKGROUNDS
            resizedMaskArr = [[None] * (N_OF_OPS)] * N_OF_BACKGROUNDS
            # print(resizedImgArr)

            resizedBg = [None] * (N_OF_BACKGROUNDS)
            bgOriginalshape = [None] * (N_OF_BACKGROUNDS)

            blur = [[None] * (N_OF_OPS)] * N_OF_BACKGROUNDS
            inv_blur = [[None] * (N_OF_OPS)] * N_OF_BACKGROUNDS
            new = [[None] * (N_OF_OPS)] * N_OF_BACKGROUNDS
            result = [[None] * (N_OF_OPS)] * N_OF_BACKGROUNDS
            resizedMaskFinal = [[None] * (N_OF_OPS)] * N_OF_BACKGROUNDS

            for bgindex in range(0, N_OF_BACKGROUNDS):
                resizedBg[bgindex], bgOriginalshape[bgindex] = loadResizedBG(
                    arrBG[i][bgindex], datasetList)

                # resizedImg = resize_with_pad(img, height=bgOriginalshape[bgindex][0], width=bgOriginalshape[bgindex][1])
                # resizedMask = resize_with_pad(mask, height=bgOriginalshape[bgindex][0], width=bgOriginalshape[bgindex][1])
                # print("bgs=",bgOriginalshape[bgindex][0],bgOriginalshape[bgindex][1])
                # print("resizedImg.shape=",resizedImg.shape)
                # print("resizedMask.shape=",resizedMask.shape)

                # calcule ops per bgs
                for opindex in range(0, N_OF_OPS):
                    globalIndex = (
                        ((realI * N_OF_BACKGROUNDS) + bgindex) * N_OF_OPS) + opindex
                    # print(globalIndex)
                    resizedImgArr[bgindex][opindex], resizedMaskArr[bgindex][opindex] = geometricOp2(
                        resizedImg, resizedMask, bgOriginalshape[bgindex], opindex, globalIndex, resizeToOrgImg(bgOriginalshape[bgindex], resizedBg[bgindex]))

                    # internalLoop
                    # BEGIN Smooth border copy

                    # disable border blur
                    # resizedMaskTmp = cv2.dilate(
                    #     resizedMaskArr[bgindex][opindex], kernelErode, iterations=1)
                    resizedMaskTmp = resizedMaskArr[bgindex][opindex]
                    blur[bgindex][opindex] = cv2.blur(resizedMaskTmp, (3, 3))

                    # blur[bgindex][opindex] = resizedMaskArr[bgindex][opindex]
                    blur[bgindex][opindex] = (
                        blur[bgindex][opindex] / 255) * 0.95
                    inv_blur[bgindex][opindex] = 1 - blur[bgindex][opindex]

                    new[bgindex][opindex] = blur[bgindex][opindex] * resizedImgArr[bgindex][opindex] + \
                        inv_blur[bgindex][opindex] * resizedBg[bgindex]
                    # END Smooth border copy

                    new[bgindex][opindex], resizedMaskArr[bgindex][opindex] = resizeToOrg(
                        bgOriginalshape[bgindex], new[bgindex][opindex], resizedMaskArr[bgindex][opindex])

                    #########################################################

                    result[bgindex][opindex] = Image.fromarray(
                        (new[bgindex][opindex]).astype(np.uint8))

                    resizedMaskFinal[bgindex][opindex] = Image.fromarray(
                        (resizedMaskArr[bgindex][opindex]).astype(np.uint8))

                    stamp = "{:06d}_{:06d}_{:03d}".format(
                        arrOBJ[i], arrBG[i][bgindex], opindex)

                    result[bgindex][opindex].save(OUTPUT_FOLDER_IMG +
                                                  "duts_image_" + stamp+ ".jpg")
                    resizedMaskFinal[bgindex][opindex].save(OUTPUT_FOLDER_MASK
                                                            + "duts_image_" + stamp+ ".png")

                    print(stamp)
                    #########################################################

            realI += 1


if __name__ == '__main__':

    if(args.n_bgs > 1):
        main(0, True, args.out_path)
    else:
        main(0, False, args.out_path)
