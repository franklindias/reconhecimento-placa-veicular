# Preprocesso.py
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math

# variáveis de nível de módulo ##########################################################################
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

###################################################################################################
def Preprocesso(imgOriginal):
    imgEscalaDeCinza = extractValue(imgOriginal)

    imgMaximoContrasteEscalaDeCinza = maximizeContrast(imgEscalaDeCinza)

    altura, largura = imgEscalaDeCinza.shape

    imgBlurred = np.zeros((altura, largura, 1), np.uint8)

    imgBlurred = cv2.GaussianBlur(imgMaximoContrasteEscalaDeCinza, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)

    imgThreshold = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    return imgEscalaDeCinza, imgThreshold
# end function

###################################################################################################
def extractValue(imgOriginal):
    altura, largura, numCanais = imgOriginal.shape

    imgHSV = np.zeros((altura, largura, 3), np.uint8)

    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

    return imgValue
# end function

###################################################################################################
def maximizeContrast(imgEscalaDeCinza):

    altura, largura = imgEscalaDeCinza.shape

    imgTopHat = np.zeros((altura, largura, 1), np.uint8)
    imgBlackHat = np.zeros((altura, largura, 1), np.uint8)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(imgEscalaDeCinza, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(imgEscalaDeCinza, cv2.MORPH_BLACKHAT, structuringElement)

    imgEscalaDeCinzaPlusTopHat = cv2.add(imgEscalaDeCinza, imgTopHat)
    imgEscalaDeCinzaPlusTopHatMinusBlackHat = cv2.subtract(imgEscalaDeCinzaPlusTopHat, imgBlackHat)

    return imgEscalaDeCinzaPlusTopHatMinusBlackHat
# end function



















