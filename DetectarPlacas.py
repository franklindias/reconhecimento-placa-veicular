# DetectarPlacas.py

import cv2
import numpy as np
import math
import Main
import random

import Preprocesso
import DetectarCaracteres
import PossivelPlaca
import PossibleCaractere

# module level variables ##########################################################################
PLACA_LARGURA_FATOR_PREENCHIMENTO = 1.3
PLACA_ALTURA_FATOR_PREENCHIMENTO = 1.5

###################################################################################################
def DetectarPlacasInScene(imgCenaOriginal):
    listaDePossiveisPlacas = []                   # this will be the return value

    altura, largura, numCanais = imgCenaOriginal.shape

    imgEscalaDeCinzaScene = np.zeros((altura, largura, 1), np.uint8)
    imgThresholdScene = np.zeros((altura, largura, 1), np.uint8)
    imgContours = np.zeros((altura, largura, 3), np.uint8)

    cv2.destroyAllWindows()

    if Main.mostrarPassos == True: # show steps #######################################################
        cv2.imshow("0", imgCenaOriginal)
    # end if # show steps #########################################################################

    imgEscalaDeCinzaScene, imgThresholdScene = Preprocesso.Preprocesso(imgCenaOriginal)         # Preprocesso to get grayscale and threshold images

    if Main.mostrarPassos == True: # show steps #######################################################
        cv2.imshow("1a", imgEscalaDeCinzaScene)
        cv2.imshow("1b", imgThresholdScene)
    # end if # show steps #########################################################################

            # find all possible chars in the scene,
            # this function first finds all contornos, then only includes contornos that could be chars (without comparison to other chars yet)
    listaDePossiveisCaracteresInScene = findPossibleCaracteresInScene(imgThresholdScene)

    if Main.mostrarPassos == True: # show steps #######################################################
        print ("step 2 - len(listaDePossiveisCaracteresInScene) = " + str(len(listaDePossiveisCaracteresInScene)))         # 131 with MCLRNF1 image

        imgContours = np.zeros((altura, largura, 3), np.uint8)

        contornos = []

        for possivelCaractere in listaDePossiveisCaracteresInScene:
            contornos.append(possivelCaractere.contour)
        # end for

        cv2.drawContours(imgContours, contornos, -1, Main.SCALAR_WHITE)
        cv2.imshow("2b", imgContours)
    # end if # show steps #########################################################################

            # given a list of all possible chars, find groups of matching chars
            # in the next steps each group of matching chars will attempt to be recognized as a Placa
    listaDeListasDeCombinacaoDeCaracteresInScene = DetectarCaracteres.findListOfListsOfMatchingCaracteres(listaDePossiveisCaracteresInScene)

    if Main.mostrarPassos == True: # show steps #######################################################
        print ("step 3 - listaDeListasDeCombinacaoDeCaracteresInScene.Count = " + str(len(listaDeListasDeCombinacaoDeCaracteresInScene)))    # 13 with MCLRNF1 image

        imgContours = np.zeros((altura, largura, 3), np.uint8)

        for listaDeCombinacaoDeCaracteres in listaDeListasDeCombinacaoDeCaracteresInScene:
            intRandomBlue = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed = random.randint(0, 255)

            contornos = []

            for matchingCaractere in listaDeCombinacaoDeCaracteres:
                contornos.append(matchingCaractere.contour)
            # end for

            cv2.drawContours(imgContours, contornos, -1, (intRandomBlue, intRandomGreen, intRandomRed))
        # end for

        cv2.imshow("3", imgContours)
    # end if # show steps #########################################################################

    for listaDeCombinacaoDeCaracteres in listaDeListasDeCombinacaoDeCaracteresInScene:                   # for each group of matching chars
        PossivelPlaca = extractPlaca(imgCenaOriginal, listaDeCombinacaoDeCaracteres)         # attempt to extract Placa

        if PossivelPlaca.imgPlaca is not None:                          # if Placa was found
            listaDePossiveisPlacas.append(PossivelPlaca)                  # add to list of possible Placas
        # end if
    # end for

    print ("\n" + str(len(listaDePossiveisPlacas)) + " possible Placas found")          # 13 with MCLRNF1 image

    if Main.mostrarPassos == True: # show steps #######################################################
        print ("\n")
        cv2.imshow("4a", imgContours)

        for i in range(0, len(listaDePossiveisPlacas)):
            p2fRectPoints = cv2.boxPoints(listaDePossiveisPlacas[i].rrLocationOfPlacaInScene)

            cv2.line(imgContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), Main.SCALAR_RED, 2)

            cv2.imshow("4a", imgContours)

            print ("possible Placa " + str(i) + ", click on any image and press a key to continue . . .")

            cv2.imshow("4b", listaDePossiveisPlacas[i].imgPlaca)
            cv2.waitKey(0)
        # end for

        print ("\nPlaca detection complete, click on any image and press a key to begin char recognition . . .\n")
        cv2.waitKey(0)
    # end if # show steps #########################################################################

    return listaDePossiveisPlacas
# end function

###################################################################################################
def findPossibleCaracteresInScene(imgThreshold):
    listaDePossiveisCaracteres = []                # this will be the return value

    intCountOfPossibleCaracteres = 0

    imgThresholdCopia = imgThreshold.copy()

    imgContours, contornos, npaHierarchy = cv2.findContours(imgThresholdCopia, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   # find all contornos

    altura, largura = imgThreshold.shape
    imgContours = np.zeros((altura, largura, 3), np.uint8)

    for i in range(0, len(contornos)):                       # for each contour

        if Main.mostrarPassos == True: # show steps ###################################################
            cv2.drawContours(imgContours, contornos, i, Main.SCALAR_WHITE)
        # end if # show steps #####################################################################

        possivelCaractere = PossibleCaractere.PossibleCaractere(contornos[i])

        if DetectarCaracteres.verificaSePossivelCaractere(possivelCaractere):                   # if contour is a possible char, note this does not compare to other chars (yet) . . .
            intCountOfPossibleCaracteres = intCountOfPossibleCaracteres + 1           # increment count of possible chars
            listaDePossiveisCaracteres.append(possivelCaractere)                        # and add to list of possible chars
        # end if
    # end for

    if Main.mostrarPassos == True: # show steps #######################################################
        print ("\nstep 2 - len(contornos) = " + str(len(contornos)))                       # 2362 with MCLRNF1 image
        print ("step 2 - intCountOfPossibleCaracteres = " + str(intCountOfPossibleCaracteres))       # 131 with MCLRNF1 image
        cv2.imshow("2a", imgContours)
    # end if # show steps #########################################################################

    return listaDePossiveisCaracteres
# end function


###################################################################################################
def extractPlaca(imgOriginal, listaDeCombinacaoDeCaracteres):
    PossivelPlaca = PossivelPlaca.PossivelPlaca()           # this will be the return value

    listaDeCombinacaoDeCaracteres.sort(key = lambda matchingCaractere: matchingCaractere.intCenterX)        # sort chars from left to right based on x position

            # calculate the center point of the Placa
    fltPlacaCenterX = (listaDeCombinacaoDeCaracteres[0].intCenterX + listaDeCombinacaoDeCaracteres[len(listaDeCombinacaoDeCaracteres) - 1].intCenterX) / 2.0
    fltPlacaCenterY = (listaDeCombinacaoDeCaracteres[0].intCenterY + listaDeCombinacaoDeCaracteres[len(listaDeCombinacaoDeCaracteres) - 1].intCenterY) / 2.0

    ptPlacaCenter = fltPlacaCenterX, fltPlacaCenterY

            # calculate Placa largura and altura
    intPlacaWidth = int((listaDeCombinacaoDeCaracteres[len(listaDeCombinacaoDeCaracteres) - 1].intBoundingRectX + listaDeCombinacaoDeCaracteres[len(listaDeCombinacaoDeCaracteres) - 1].intBoundingRectWidth - listaDeCombinacaoDeCaracteres[0].intBoundingRectX) * PLACA_LARGURA_FATOR_PREENCHIMENTO)

    intTotalOfCaractereHeights = 0

    for matchingCaractere in listaDeCombinacaoDeCaracteres:
        intTotalOfCaractereHeights = intTotalOfCaractereHeights + matchingCaractere.intBoundingRectHeight
    # end for

    fltAverageCaractereHeight = intTotalOfCaractereHeights / len(listaDeCombinacaoDeCaracteres)

    intPlacaHeight = int(fltAverageCaractereHeight * PLACA_ALTURA_FATOR_PREENCHIMENTO)

            # calculate correction angle of Placa region
    fltOpposite = listaDeCombinacaoDeCaracteres[len(listaDeCombinacaoDeCaracteres) - 1].intCenterY - listaDeCombinacaoDeCaracteres[0].intCenterY
    fltHypotenuse = DetectarCaracteres.distanciaEntreCaracteres(listaDeCombinacaoDeCaracteres[0], listaDeCombinacaoDeCaracteres[len(listaDeCombinacaoDeCaracteres) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

            # pack Placa region center point, largura and altura, and correction angle into rotated rect member variable of Placa
    PossivelPlaca.rrLocationOfPlacaInScene = ( tuple(ptPlacaCenter), (intPlacaWidth, intPlacaHeight), fltCorrectionAngleInDeg )

            # final steps are to perform the actual rotation

            # get the rotation matrix for our calculated correction angle
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlacaCenter), fltCorrectionAngleInDeg, 1.0)

    altura, largura, numCanais = imgOriginal.shape      # unpack original image largura and altura

    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (largura, altura))       # rotate the entire image

    imgCropped = cv2.getRectSubPix(imgRotated, (intPlacaWidth, intPlacaHeight), tuple(ptPlacaCenter))

    PossivelPlaca.imgPlaca = imgCropped         # copy the cropped Placa image into the applicable member variable of the possible Placa

    return PossivelPlaca
# end function












