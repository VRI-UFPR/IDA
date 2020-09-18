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

from argparse import ArgumentParser
# import res2net
import os
import cv2
import numpy as np
from PIL import Image
import time
# import scipy
from sklearn.neighbors import NearestNeighbors
import skimage.feature as ft


import torch

import sys
# sys.path.append('./res2net/')

# for fancy parameterization


def parse_args():
    parser = ArgumentParser(
        description='Compute feature vectors for the objects and backgrounds for the duts dataset')

    parser.add_argument(
        '-device', '--device',
        type=str, default="cpu",
        help='DEVICE values "cpu","cuda" '
    )

    parser.add_argument(
        '-dataset_file', '--dataset_file',
        type=str, default="/home/bakrinski/workspace/datasets/duts/training/paths.txt",
        help='DATASET_FILE'
    )

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
        '-metric_knn', '--metric_knn',
        type=str, default="cosine",
        help='distance function used in the knn'
    )

    parser.add_argument(
        '-nbins', '--nbins',
        type=int, default=64,
        help='number of bins for each histogram channel'
    )

    parser.add_argument(
        '-size', '--size',
        type=int, default=10553,
        help='number of images in the dataset'
    )

    parser.add_argument(
        '-k', '--k',
        type=int, default=10553,
        help='number of k NearestNeighbors'
    )

    return parser.parse_args()


# CALL PARSER
args = parse_args()
#

# SETTINGS
OBJ_FOLDER_IMG = args.obj_path
OBJ_FOLDER_MASK = args.obj_mask_path
BG_FOLDER_IMG = args.bg_path
NBINS = args.nbins
DATASET_SIZE = args.size
N_NEIGHBORS = args.k
METRIC_KNN = args.metric_knn
DEVICE = args.device
DATASET_FILE = args.dataset_file
##


def getHistograms(img):
    imgHsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(imgHsv)

    histH, _ = np.histogram(h, bins=NBINS, density=True)
    histS, _ = np.histogram(s, bins=NBINS, density=True)
    histV, _ = np.histogram(v, bins=NBINS, density=True)

    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # settings for LBP
    # radius = 3
    # n_points = 8 * radiusDATASET_SIZE
    # METHOD = 'uniform'
    # lbp = ft.local_binary_pattern(imgGray, n_points, radius, METHOD)
    lbp = ft.local_binary_pattern(imgGray, 24, 3, 'uniform')

    histLBP, _ = np.histogram(lbp, bins=NBINS, density=True)

    hist = np.concatenate((histH, histS, histV, histLBP))

    return hist


def getHistogramsWithMask(img, mask):
    imgHsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(imgHsv)

    histH, _ = np.histogram(h, bins=NBINS, density=True, weights=mask)
    histS, _ = np.histogram(s, bins=NBINS, density=True, weights=mask)
    histV, _ = np.histogram(v, bins=NBINS, density=True, weights=mask)

    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # settings for LBP
    # radius = 3
    # n_points = 8 * radius
    # METHOD = 'uniform'
    # lbp = ft.local_binary_pattern(imgGray, n_points, radius, METHOD)
    lbp = ft.local_binary_pattern(imgGray, 24, 3, 'uniform')

    # fastlbp = fast_lbp(img, device, padding_type)
    # fastlbp = fast_lbp(imgGray, "cuda", 0)
    # fastlbp = fast_lbp(imgGray, "cpu", 0)

    # print("lbp:",lbp)
    # print("fastlbp:",fastlbp)

    histLBP, _ = np.histogram(lbp, bins=NBINS, density=True, weights=mask)

    hist = np.concatenate((histH, histS, histV, histLBP))

    return hist


def getHistogramsTorch(img):

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

    # fd, hog_image = ft.hog(imgGray, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False, multichannel=False, feature_vector=True)

    # print("hog_image.shape=",hog_image.shape)
    # exit(0)

    hist = torch.stack((histH, histS, histV, histLBP))
    hist = hist.view(-1)

    return hist


def getHistogramsWithMaskTorch(img, mask):

    if img.mode == "L":
        img = img.convert('RGB')

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

    # hog_result = ft.hog(imgGray, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False, multichannel=False, feature_vector=True)
    # hog_result = ft.hog(imgGray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys', visualize=False, transform_sqrt=False, feature_vector=True, multichannel=None)
    # print("hog_result",hog_result)
    # print("hog_result.shape",hog_result.shape)

    # histHOG = torch.histc(torch.from_numpy(hog_result).float().to(DEVICE), bins=NBINS*4, min=0.0, max=255.0)

    # print("histHOG.shape=",histHOG.shape)
    # exit(0)

    hist = torch.stack((histH, histS, histV, histLBP))
    hist = hist.view(-1)

    return hist


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


def main():

    datasetFile = DATASET_FILE
    datasetList = load_list(datasetFile)

    existsDataSetFile = os.path.isfile('dataset.txt')
    if not(existsDataSetFile):
        with open('dataset.txt', 'w') as fd:
            for i in range(0, DATASET_SIZE):
                print(i, file=fd)

    print("now obj")
    existsObj = os.path.isfile('histogramsOBJ.npy')
    if not(existsObj):
        print("building histograms")

        histogramsOBJ = np.empty((DATASET_SIZE, NBINS * 4), np.float32)
        for i in range(0, DATASET_SIZE, 1):

            imgName = datasetList[i][0]
            imFile = Image.open(OBJ_FOLDER_IMG + imgName)
            img = imFile

            maskName = datasetList[i][1]

            maskFile = Image.open(OBJ_FOLDER_MASK + maskName)
            maskFile = maskFile.convert('L')
            mask = np.array(maskFile) / 255.

            hist = getHistogramsWithMaskTorch(img, mask)

            histogramsOBJ[i] = hist

            imFile.close()
            maskFile.close()
            # print(i)
        print("saving array histogramsOBJ.npy")
        np.save("histogramsOBJ.npy", histogramsOBJ)
    else:
        print("loading array histogramsOBJ.npy")
        histogramsOBJ = np.load("histogramsOBJ.npy")

    print("now bg")
    existsBg = os.path.isfile('histogramsBG.npy')
    if not(existsBg):
        print("building histograms")
        histogramsBG = np.empty((DATASET_SIZE, NBINS * 4), np.float32)
        for i in range(0, DATASET_SIZE):
            bgName = datasetList[i][0]
            ##special case
            bgName= bgName.replace(".jpg","_inpaint.jpg")

            bgFile = Image.open(BG_FOLDER_IMG + bgName)
            bg = bgFile

            hist = getHistogramsTorch(bg)

            histogramsBG[i] = hist

            bgFile.close()

        print("saving array histogramsBG.npy")
        np.save("histogramsBG.npy", histogramsBG)
    else:
        print("loading array histogramsBG.npy")
        histogramsBG = np.load("histogramsBG.npy")

    print("now knn")
    if( os.path.isfile("distancesKnn_"+METRIC_KNN+".npy" )):
        print("loading array distancesKnn_"+METRIC_KNN+".npy")
        print("loading array indicesKnn_"+METRIC_KNN+".npy")
        distances = np.load("distancesKnn_"+METRIC_KNN+".npy")
        indices = np.load("indicesKnn_"+METRIC_KNN+".npy")
    else:
        nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORS, metric=METRIC_KNN,
                                algorithm='auto', n_jobs=-1).fit(histogramsBG)

        distances, indices = nbrs.kneighbors(histogramsOBJ)

        print("saving array distancesKnn_"+METRIC_KNN+".npy")
        print("saving array indicesKnn_"+METRIC_KNN+".npy")
        np.save("distancesKnn_"+METRIC_KNN+".npy", distances)
        np.save("indicesKnn_"+METRIC_KNN+".npy", indices)

    HALF_N_NEIGHBORS = int(np.floor(N_NEIGHBORS / 2))

    #
    # print(indices[1979][HALF_N_NEIGHBORS],distances[1979][HALF_N_NEIGHBORS])
    # print(indices[1078][HALF_N_NEIGHBORS],distances[1078][HALF_N_NEIGHBORS])
    # exit()
    #
    print("creating",'distances_' + METRIC_KNN + '.txt', "and", 'indices_' + METRIC_KNN + '.txt')
    with open('distances_' + METRIC_KNN + '.txt', 'w') as fd:
        with open('indices_' + METRIC_KNN + '.txt', 'w') as fi:
            for i in range(0, N_NEIGHBORS):
                # np.median()

                std = torch.std(distances[i])
                total = torch.sum(distances[i])
                mean = total / N_NEIGHBORS
                closest_distance = distances[i][HALF_N_NEIGHBORS]
                closest_index = HALF_N_NEIGHBORS
                for j in range(0, N_NEIGHBORS):
                    if(np.abs(mean+(std) - distances[i][j]) < np.abs(mean+(std) - closest_distance)):
                        closest_index = j
                        closest_distance = distances[i][j]

                # print(i,"closest_index=",closest_index)
                # print(i,"mean=",mean)
                # print(i,"std=",std)
                # print(i,"mean+(std)=",mean+(std))
                valuesDis = str(distances[i][closest_index])
                valuesIndex = str(indices[i][closest_index])

                # valuesDis = str(distances[i][HALF_N_NEIGHBORS])
                # valuesIndex = str(indices[i][HALF_N_NEIGHBORS])

                print(valuesDis, file=fd)
                print(valuesIndex, file=fi)


main()
