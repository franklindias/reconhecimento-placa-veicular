# DetectarCaracteres.py

import cv2
import numpy as np
import math
import random

import Main
import Preprocesso
import PossivelCaractere

# module level variables ##########################################################################

kNearest = cv2.ml.KNearest_create()

# constants for verificaSePossivelCaractere, esta verifica apenas um caractere possivel (nao compara a outro caractere)
MIN_PIXEL_LARGURA = 2
MIN_PIXEL_ALTURA = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 80

# Constantes para comparar dois caracteres
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_MUDANCA_EM_AREA = 0.5

MAX_MUDANCA_NA_LARGURA = 0.8
MAX_MUDANCA_NA_ALTURA = 0.2

MAX_ANGULO_ENTRE_CARACTERES = 12.0

        # outras constantes
MIN_NUMERO_DE_COMBINACAO_CARACTERES = 3

REDIMENSIONAR_CHAR_IMAGEM_LARGURA = 20
REDIMENSIONAR_CHAR_IMAGEM_ALTURA = 30

MIN_CONTORNO_AREA = 100

###################################################################################################
def loadKNNDataAndTrainKNN():

    todosOsContornosComOsDados = []                # declare empty lists,
    contornosValidosComData = []              # we will fill these shortly

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)                  
        # read in training classifications
    except:                                                                                 
    # if file could not be opened
        print ("error, unable to open classifications.txt, exiting program\n")                
        # show error message
        os.system("pause")
        return False                                                                        
        # and return False
    # end try

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)                 
        # read in training images
    except:                                                                                 
    # if file could not be opened
        print ("error, unable to open flattened_images.txt, exiting program\n")               
        # show error message
        os.system("pause")
        return False                                                                        
        # and return False
    # end try

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       
    # reshape numpy array to 1d, necessary to pass to call to train

    kNearest.setDefaultK(1)                                                             
    # set default K to 1

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)           
    # train KNN object

    return True                             
    # if we got here training was successful so return true
# end function

###################################################################################################
def DetectarCaracteresNasPlacas(listaDePossiveisPlacas):
    intPlacaCounter = 0
    imgContours = None
    contornos = []

    if len(listaDePossiveisPlacas) == 0:          
    # if list of possible Placas is empty
        return listaDePossiveisPlacas             
        # return
    # end if

            
    # at this point we can be sure the list of possible Placas has at least one Placa

    for PossivelPlaca in listaDePossiveisPlacas:         
    # for each possible Placa, this is a big for loop that takes up most of the function

        PossivelPlaca.imgEscalaDeCinza, PossivelPlaca.imgThreshold = Preprocesso.Preprocesso(PossivelPlaca.imgPlaca)     
        # Preprocesso to get grayscale and threshold images

        if Main.mostrarPassos == True: # show steps ###################################################
            cv2.imshow("5a", PossivelPlaca.imgPlaca)
            cv2.imshow("5b", PossivelPlaca.imgEscalaDeCinza)
            cv2.imshow("5c", PossivelPlaca.imgThreshold)
        # end if # show steps #####################################################################

                # increase size of Placa image for easier viewing and char detection
        PossivelPlaca.imgThreshold = cv2.resize(PossivelPlaca.imgThreshold, (0, 0), fx = 1.6, fy = 1.6)

                # threshold again to eliminate any gray areas
        thresholdValue, PossivelPlaca.imgThreshold = cv2.threshold(PossivelPlaca.imgThreshold, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if Main.mostrarPassos == True: # show steps ###################################################
            cv2.imshow("5d", PossivelPlaca.imgThreshold)
        # end if # show steps #####################################################################

                # find all possible chars in the Placa,
                # this function first finds all contornos, then only includes contornos that could be chars (without comparison to other chars yet)
        listaDePossiveisCaracteresInPlaca = encontrarPossivelCaractereNaPlaca(PossivelPlaca.imgEscalaDeCinza, PossivelPlaca.imgThreshold)

        if Main.mostrarPassos == True: # show steps ###################################################
            altura, largura, numCanais = PossivelPlaca.imgPlaca.shape
            imgContours = np.zeros((altura, largura, 3), np.uint8)
            del contornos[:]                                         # clear the contornos list

            for possivelCaractere in listaDePossiveisCaracteresInPlaca:
                contornos.append(possivelCaractere.contour)
            # end for

            cv2.drawContours(imgContours, contornos, -1, Main.SCALAR_WHITE)

            cv2.imshow("6", imgContours)
        # end if # show steps #####################################################################

                # given a list of all possible chars, find groups of matching chars within the Placa
        listaDeListasDeCombinacaoDeCaracteresInPlaca = findListOfListsOfMatchingCaracteres(listaDePossiveisCaracteresInPlaca)

        if Main.mostrarPassos == True: # show steps ###################################################
            imgContours = np.zeros((altura, largura, 3), np.uint8)
            del contornos[:]

            for listaDeCombinacaoDeCaracteres in listaDeListasDeCombinacaoDeCaracteresInPlaca:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)

                for matchingCaractere in listaDeCombinacaoDeCaracteres:
                    contornos.append(matchingCaractere.contour)
                # end for
                cv2.drawContours(imgContours, contornos, -1, (intRandomBlue, intRandomGreen, intRandomRed))
            # end for
            cv2.imshow("7", imgContours)
        # end if # show steps #####################################################################

        if (len(listaDeListasDeCombinacaoDeCaracteresInPlaca) == 0):			# if no groups of matching chars were found in the Placa

            if Main.mostrarPassos == True: # show steps ###############################################
                print ("chars found in Placa number " + str(intPlacaCounter) + " = (none), click on any image and press a key to continue . . .")
                intPlacaCounter = intPlacaCounter + 1
                cv2.destroyWindow("8")
                cv2.destroyWindow("9")
                cv2.destroyWindow("10")
                cv2.waitKey(0)
            # end if # show steps #################################################################

            PossivelPlaca.strCaracteres = ""
            continue						# go back to top of for loop
        # end if

        for i in range(0, len(listaDeListasDeCombinacaoDeCaracteresInPlaca)):                              # within each list of matching chars
            listaDeListasDeCombinacaoDeCaracteresInPlaca[i].sort(key = lambda matchingCaractere: matchingCaractere.intCenterX)        # sort chars from left to right
            listaDeListasDeCombinacaoDeCaracteresInPlaca[i] = removerSobreposicaoDeCaracteres(listaDeListasDeCombinacaoDeCaracteresInPlaca[i])              # and remove inner overlapping chars
        # end for

        if Main.mostrarPassos == True: # show steps ###################################################
            imgContours = np.zeros((altura, largura, 3), np.uint8)

            for listaDeCombinacaoDeCaracteres in listaDeListasDeCombinacaoDeCaracteresInPlaca:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)

                del contornos[:]

                for matchingCaractere in listaDeCombinacaoDeCaracteres:
                    contornos.append(matchingCaractere.contour)
                # end for

                cv2.drawContours(imgContours, contornos, -1, (intRandomBlue, intRandomGreen, intRandomRed))
            # end for
            cv2.imshow("8", imgContours)
        # end if # show steps #####################################################################

                # within each possible Placa, suppose the longest list of potential matching chars is the actual list of chars
        intLenOfLongestListOfCaracteres = 0
        intIndexOfLongestListOfCaracteres = 0

                # loop through all the vectors of matching chars, get the index of the one with the most chars
        for i in range(0, len(listaDeListasDeCombinacaoDeCaracteresInPlaca)):
            if len(listaDeListasDeCombinacaoDeCaracteresInPlaca[i]) > intLenOfLongestListOfCaracteres:
                intLenOfLongestListOfCaracteres = len(listaDeListasDeCombinacaoDeCaracteresInPlaca[i])
                intIndexOfLongestListOfCaracteres = i
            # end if
        # end for

                # suppose that the longest list of matching chars within the Placa is the actual list of chars
        
        maiorListaDeCaracteresCorrespondentesNaPlaca = listaDeListasDeCombinacaoDeCaracteresInPlaca[intIndexOfLongestListOfCaracteres]

        if Main.mostrarPassos == True: # show steps ###################################################
            imgContours = np.zeros((altura, largura, 3), np.uint8)
            del contornos[:]

            for matchingCaractere in maiorListaDeCaracteresCorrespondentesNaPlaca:
                contornos.append(matchingCaractere.contour)
            # end for

            cv2.drawContours(imgContours, contornos, -1, Main.SCALAR_WHITE)

            cv2.imshow("9", imgContours)
        # end if # show steps #####################################################################

        PossivelPlaca.strCaracteres = recognizeCaracteresInPlaca(PossivelPlaca.imgThreshold, 
    maiorListaDeCaracteresCorrespondentesNaPlaca)

        if Main.mostrarPassos == True: # show steps ###################################################
            print ("Caracteres encontrados no numero da placa " + str(intPlacaCounter) + " = " + PossivelPlaca.strCaracteres + ", click on any image and press a key to continue . . .")
            intPlacaCounter = intPlacaCounter + 1
            cv2.waitKey(0)
        # end if # show steps #####################################################################

    # end of big for loop that takes up most of the function

    if Main.mostrarPassos == True:
        print ("\nDeteccaoo de caracteres completa, clique em qualquer imagem e pressione uma tecla para continuar . . .\n")
        cv2.waitKey(0)
    # end if

    return listaDePossiveisPlacas
# end function

###################################################################################################
def encontrarPossivelCaractereNaPlaca(imgEscalaDeCinza, imgThreshold):
    listaDePossiveisCaracteres = []                        # this will be the return value
    contornos = []
    imgThresholdCopia = imgThreshold.copy()

            # find all contornos in Placa
    imgContours, contornos, npaHierarchy = cv2.findContours(imgThresholdCopia, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contornos:                        # for each contour
        possivelCaractere = PossibleCaractere.PossibleCaractere(contour)

        if verificaSePossivelCaractere(possivelCaractere):              # if contour is a possible char, note this does not compare to other chars (yet) . . .
            listaDePossiveisCaracteres.append(possivelCaractere)       # add to list of possible chars
        # end if
    # end if

    return listaDePossiveisCaracteres
# end function

###################################################################################################
def verificaSePossivelCaractere(possivelCaractere):
            # this function is a 'first pass' that does a rough check on a contour to see if it could be a char,
            # note that we are not (yet) comparing the char to other chars to look for a group
    if (possivelCaractere.intBoundingRectArea > MIN_PIXEL_AREA and
        possivelCaractere.intBoundingRectWidth > MIN_PIXEL_LARGURA and possivelCaractere.intBoundingRectHeight > MIN_PIXEL_ALTURA and
        MIN_ASPECT_RATIO < possivelCaractere.fltAspectRatio and possivelCaractere.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False
    # end if
# end function

###################################################################################################
def findListOfListsOfMatchingCaracteres(listaDePossiveisCaracteres):
            # with this function, we start off with all the possible chars in one big list
            # the purpose of this function is to re-arrange the one big list of chars into a list of lists of matching chars,
            # note that chars that are not found to be in a group of matches do not need to be considered further
    listaDeListasDeCombinacaoDeCaracteres = []                  
    # this will be the return value

    for possivelCaractere in listaDePossiveisCaracteres:                        
    # for each possible char in the one big list of chars
        listaDeCombinacaoDeCaracteres = encontrarListaDeCombincacaoDeCaracteres(possivelCaractere, listaDePossiveisCaracteres)        
        # find all chars in the big list that match the current char

        listaDeCombinacaoDeCaracteres.append(possivelCaractere)                
        # also add the current char to current possible list of matching chars

        if len(listaDeCombinacaoDeCaracteres) < MIN_NUMERO_DE_COMBINACAO_CARACTERES:     
        # if current possible list of matching chars is not long enough to constitute a possible Placa
            continue                            
            # jump back to the top of the for loop and try again with next char, note that it's not necessary
                                                
            # to save the list in any way since it did not have enough chars to be a possible Placa
        # end if

        # if we get here, the current list passed test as a "group" or "cluster" of matching chars
        listaDeListasDeCombinacaoDeCaracteres.append(listaDeCombinacaoDeCaracteres)      
        # so add to our list of lists of matching chars

        listaDePossiveisCaracteresComAtualCombinacaoRemovida = []

                                                # remove the current list of matching chars from the big list so we don't use those same chars twice,
                                                # make sure to make a new big list for this since we don't want to change the original big list
        listaDePossiveisCaracteresComAtualCombinacaoRemovida = list(set(listaDePossiveisCaracteres) - set(listaDeCombinacaoDeCaracteres))

        recursiveListOfListsOfMatchingCaracteres = findListOfListsOfMatchingCaracteres(listaDePossiveisCaracteresComAtualCombinacaoRemovida)      
        # recursive call

        for listaRecursivaDeCombinacaoDeCaracteres in recursiveListOfListsOfMatchingCaracteres:        
        # for each list of matching chars found by recursive call
            listaDeListasDeCombinacaoDeCaracteres.append(listaRecursivaDeCombinacaoDeCaracteres)             
            # add to our original list of lists of matching chars
        # end for

        break       # exit for

    # end for

    return listaDeListasDeCombinacaoDeCaracteres
# end function

###################################################################################################
def encontrarListaDeCombincacaoDeCaracteres(possivelCaractere, listOfCaracteres):
            # the purpose of this function is, given a possible char and a big list of possible chars,
            # find all chars in the big list that are a match for the single possible char, and return those matching chars as a list

    listaDeCombinacaoDeCaracteres = []                
    # this will be the return value

    for possivelCombinacaoDeCaractere in listOfCaracteres:                
    # for each char in big list
        if possivelCombinacaoDeCaractere == possivelCaractere:    
        # if the char we attempting to find matches for is the exact same char as the char in the big list we are currently checking
        # then we should not include it in the list of matches b/c that would end up double including the current char
            continue                                
            # so do not add to list of matches and jump back to top of for loop
        # end if
                    # compute stuff to see if chars are a match
        fltDistanciaEntreCaracteres = distanciaEntreCaracteres(possivelCaractere, possivelCombinacaoDeCaractere)

        fltAnguloEntreCaracteres = anguloEntreCaracteres(possivelCaractere, possivelCombinacaoDeCaractere)

        fltChangeInArea = float(abs(possivelCombinacaoDeCaractere.intBoundingRectArea - possivelCaractere.intBoundingRectArea)) / float(possivelCaractere.intBoundingRectArea)

        fltChangeInWidth = float(abs(possivelCombinacaoDeCaractere.intBoundingRectWidth - possivelCaractere.intBoundingRectWidth)) / float(possivelCaractere.intBoundingRectWidth)
        fltChangeInHeight = float(abs(possivelCombinacaoDeCaractere.intBoundingRectHeight - possivelCaractere.intBoundingRectHeight)) / float(possivelCaractere.intBoundingRectHeight)

                # check if chars match
        if (fltDistanciaEntreCaracteres < (possivelCaractere.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
            fltAnguloEntreCaracteres < MAX_ANGULO_ENTRE_CARACTERES and
            fltChangeInArea < MAX_MUDANCA_EM_AREA and
            fltChangeInWidth < MAX_MUDANCA_NA_LARGURA and
            fltChangeInHeight < MAX_MUDANCA_NA_ALTURA):

            listaDeCombinacaoDeCaracteres.append(possivelCombinacaoDeCaractere)        
            # if the chars are a match, add the current char to list of matching chars
        # end if
    # end for

    return listaDeCombinacaoDeCaracteres                  
    # return result
# end function

###################################################################################################
# use Pythagorean theorem to calculate distance between two chars
def distanciaEntreCaracteres(primeiroCaractere, segundoCaractere):
    intX = abs(primeiroCaractere.intCenterX - segundoCaractere.intCenterX)
    intY = abs(primeiroCaractere.intCenterY - segundoCaractere.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))
# end function

###################################################################################################
# use basic trigonometry (SOH CAH TOA) to calculate angle between chars
def anguloEntreCaracteres(primeiroCaractere, segundoCaractere):
    fltAdj = float(abs(primeiroCaractere.intCenterX - segundoCaractere.intCenterX))
    fltOpp = float(abs(primeiroCaractere.intCenterY - segundoCaractere.intCenterY))

    if fltAdj != 0.0:                           
    # check to make sure we do not divide by zero if the center X positions are equal, float division by zero will cause a crash in Python
        fltAngleInRad = math.atan(fltOpp / fltAdj)      
        # if adjacent is not zero, calculate angle
    else:
        fltAngleInRad = 1.5708                          
        # if adjacent is zero, use this as the angle, this is to be consistent with the C++ version of this program
    # end if

    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)       
    # calculate angle in degrees

    return fltAngleInDeg
# end function

###################################################################################################
# if we have two chars overlapping or to close to each other to possibly be separate chars, remove the inner (smaller) char,
# this is to prevent including the same char twice if two contornos are found for the same char,
# for example for the letter 'O' both the inner ring and the outer ring may be found as contornos, but we should only include the char once
def removerSobreposicaoDeCaracteres(listaDeCombinacaoDeCaracteres):
    listaDeCombinacaoDeCaracteresWithInnerCaractereRemoved = list(listaDeCombinacaoDeCaracteres)                
    # this will be the return value

    for caractereAtual in listaDeCombinacaoDeCaracteres:
        for outroCaractere in listaDeCombinacaoDeCaracteres:
            if caractereAtual != outroCaractere:        
            # if current char and other char are not the same char . . .
                                                                            
                # if current char and other char have center points at almost the same location . . .
                if distanciaEntreCaracteres(caractereAtual, outroCaractere) < (caractereAtual.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                                # if we get in here we have found overlapping chars
                                # next we identify which char is smaller, then if that char was not already removed on a previous pass, remove it
                    if caractereAtual.intBoundingRectArea < outroCaractere.intBoundingRectArea:         
                    # if current char is smaller than other char
                        if caractereAtual in listaDeCombinacaoDeCaracteresWithInnerCaractereRemoved:              
                        # if current char was not already removed on a previous pass . . .
                            listaDeCombinacaoDeCaracteresWithInnerCaractereRemoved.remove(caractereAtual)         
                            # then remove current char
                        # end if
                    else:                                                                       
                    # else if other char is smaller than current char
                        if outroCaractere in listaDeCombinacaoDeCaracteresWithInnerCaractereRemoved:                
                        # if other char was not already removed on a previous pass . . .
                            listaDeCombinacaoDeCaracteresWithInnerCaractereRemoved.remove(outroCaractere)           
                            # then remove other char
                        # end if
                    # end if
                # end if
            # end if
        # end for
    # end for

    return listaDeCombinacaoDeCaracteresWithInnerCaractereRemoved
# end function

###################################################################################################
# this is where we apply the actual char recognition
def recognizeCaracteresInPlaca(imgThreshold, listaDeCombinacaoDeCaracteres):
    strCaracteres = ""               # this will be the return value, the chars in the lic Placa

    altura, largura = imgThreshold.shape

    imgThresholdColor = np.zeros((altura, largura, 3), np.uint8)

    listaDeCombinacaoDeCaracteres.sort(key = lambda matchingCaractere: matchingCaractere.intCenterX)        
    # sort chars from left to right

    cv2.cvtColor(imgThreshold, cv2.COLOR_GRAY2BGR, imgThresholdColor)                     
    # make color version of threshold image so we can draw contornos in color on it

    for caractereAtual in listaDeCombinacaoDeCaracteres:                                         
    # for each char in Placa
        pt1 = (caractereAtual.intBoundingRectX, caractereAtual.intBoundingRectY)
        pt2 = ((caractereAtual.intBoundingRectX + caractereAtual.intBoundingRectWidth), (caractereAtual.intBoundingRectY + caractereAtual.intBoundingRectHeight))

        cv2.rectangle(imgThresholdColor, pt1, pt2, Main.SCALAR_GREEN, 2)           
        # draw green box around the char

                # crop char out of threshold image
        imgROI = imgThreshold[caractereAtual.intBoundingRectY : caractereAtual.intBoundingRectY + caractereAtual.intBoundingRectHeight,
                           caractereAtual.intBoundingRectX : caractereAtual.intBoundingRectX + caractereAtual.intBoundingRectWidth]

        imgROIResized = cv2.resize(imgROI, (REDIMENSIONAR_CHAR_IMAGEM_LARGURA, REDIMENSIONAR_CHAR_IMAGEM_ALTURA))           # resize image, this is necessary for char recognition

        npaROIResized = imgROIResized.reshape((1, REDIMENSIONAR_CHAR_IMAGEM_LARGURA * REDIMENSIONAR_CHAR_IMAGEM_ALTURA))        # flatten image into 1d numpy array

        npaROIResized = np.float32(npaROIResized)               
        # convert from 1d numpy array of ints to 1d numpy array of floats

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)              # finally we can call findNearest !!!

        strCurrentCaractere = str(chr(int(npaResults[0][0])))            
        # get character from results

        strCaracteres = strCaracteres + strCurrentCaractere                        
        # append current char to full string

    # end for

    if Main.mostrarPassos == True: # show steps #######################################################
        cv2.imshow("10", imgThresholdColor)
    # end if # show steps #########################################################################

    return strCaracteres
# end function








