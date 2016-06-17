# Main.py

import cv2
import numpy as np
import os

import DetectarCaracteres
import DetectarPlacas
import PossivelPlaca

# module level variables ##########################################################################
ESCALA_PRETO = (0.0, 0.0, 0.0)
ESCALA_BRANCO = (255.0, 255.0, 255.0)
ESCALA_AMARELO = (0.0, 255.0, 255.0)
ESCALA_VERDE = (0.0, 255.0, 0.0)
ESCALA_VERMELHO = (0.0, 0.0, 255.0)

mostrarPassos = False


###################################################################################################
def main():

    blnKNNTrainingSuccessful = DetectarCaracteres.loadKNNDataAndTrainKNN()         
    # attempt KNN training

    if blnKNNTrainingSuccessful == False:                               
    # if KNN training was not successful
        print ("\nerror: KNN traning was not successful\n")               
        # show error message
        return                                                          
        # and exit program
    # end if

    imgCenaOriginal  = cv2.imread("imagens/1.png")               
    # open image

    if imgCenaOriginal is None:                            
    # if image was not read successfully
        print ("\nErro: Imagem nao lida do arquivo \n\n")      
        # print error message to std out
        os.system("pause")                                  
        # pause so user can see error message
        return                                              
        # and exit program
    # end if

    listaDePossiveisPlacas = DetectarPlacas.DetectarPlacasInScene(imgCenaOriginal)           
    # detect Placas

    listaDePossiveisPlacas = DetectarCaracteres.DetectarCaracteresNasPlacas(listaDePossiveisPlacas)        
    # detect chars in Placas

    cv2.imshow("imgCenaOriginal", imgCenaOriginal)            
    # show scene image

    if len(listaDePossiveisPlacas) == 0:                          
    # if no Placas were found
        print ("\nNenhuma placa foi encontrada\n")            
        # inform user no Placas were found
    else:                                                       
    # else
                
                # if we get in here list of possible Placas has at leat one Placa

                # sort the list of possible Placas in DESCENDING order (most number of chars to least number of chars)
        listaDePossiveisPlacas.sort(key = lambda possivelPlaca: len(possivelPlaca.strCaracteres), reverse = True)

                # suppose the Placa with the most recognized chars (the first Placa in sorted by string length descending order) is the actual Placa
        licPlaca = listaDePossiveisPlacas[0]

        cv2.imshow("imgPlaca", licPlaca.imgPlaca)           
        # show crop of Placa and threshold of Placa
        cv2.imshow("imgThreshold", licPlaca.imgThreshold)

        if len(licPlaca.strCaracteres) == 0:                    
        # if no chars were found in the Placa
            print ("\nNenhum caractere foi encontrado\n\n")       
            # show message
            return                                          
            # and exit program
        # end if

        desenharRetanguloVermelhoAoRedorDaPlaca(imgCenaOriginal, licPlaca)            
        # draw red rectangle around Placa

        print ("\nPlca lida da imagem = " + licPlaca.strCaracteres + "\n")       
        # write license Placa text to std out
        print ("----------------------------------------")

        escreverCaracteresDaPlacaNaImagem(imgCenaOriginal, licPlaca)           
        # write license Placa text on the image

        cv2.imshow("imgCenaOriginal", imgCenaOriginal)                
        # re-show scene image

        cv2.imwrite("imgCenaOriginal.png", imgCenaOriginal)           
        # write image out to file

    # end if else

    cv2.waitKey(0)					
    # hold windows open until user presses a key

    return
# end main

###################################################################################################
def desenharRetanguloVermelhoAoRedorDaPlaca(imgCenaOriginal, licPlaca):

    p2fRectPoints = cv2.boxPoints(licPlaca.rrLocationOfPlacaInScene)            
    # get 4 vertices of rotated rect

    cv2.line(imgCenaOriginal, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), ESCALA_VERMELHO, 2)         
    # draw 4 red lines
    cv2.line(imgCenaOriginal, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), ESCALA_VERMELHO, 2)
    cv2.line(imgCenaOriginal, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), ESCALA_VERMELHO, 2)
    cv2.line(imgCenaOriginal, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), ESCALA_VERMELHO, 2)
# end function

###################################################################################################
def escreverCaracteresDaPlacaNaImagem(imgCenaOriginal, licPlaca):
    ptCenterOfTextAreaX = 0                             
    # this will be the center of the area the text will be written to
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0                          
    # this will be the bottom left of the area that the text will be written to
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgCenaOriginal.shape
    PlacaHeight, PlacaWidth, PlacaNumChannels = licPlaca.imgPlaca.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX                      
    # choose a plain jane font
    fltFontScale = float(PlacaHeight) / 30.0                    
    # base font scale on altura of Placa area
    intFontThickness = int(round(fltFontScale * 1.5))           
    # base font thickness on font scale

    textSize, baseline = cv2.getTextSize(licPlaca.strCaracteres, intFontFace, fltFontScale, intFontThickness)        
    # call getTextSize

            # unpack roatated rect into center point, largura and altura, and angle
    ( (intPlacaCenterX, intPlacaCenterY), (intPlacaWidth, intPlacaHeight), fltCorrectionAngleInDeg ) = licPlaca.rrLocationOfPlacaInScene

    intPlacaCenterX = int(intPlacaCenterX)              
    # make sure center is an integer
    intPlacaCenterY = int(intPlacaCenterY)

    ptCenterOfTextAreaX = int(intPlacaCenterX)         
    # the horizontal location of the text area is the same as the Placa

    if intPlacaCenterY < (sceneHeight * 0.75):                                                  
    # if the license Placa is in the upper 3/4 of the image
        ptCenterOfTextAreaY = int(round(intPlacaCenterY)) + int(round(PlacaHeight * 1.6))      
        # write the chars in below the Placa
    else:                                                                                       
    # else if the license Placa is in the lower 1/4 of the image
        ptCenterOfTextAreaY = int(round(intPlacaCenterY)) - int(round(PlacaHeight * 1.6))      
        # write the chars in above the Placa
    # end if

    textSizeWidth, textSizeHeight = textSize                
    # unpack text size largura and altura

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))           
    # calculate the lower left origin of the text area
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))          
    # based on the text area center, largura, and altura

            # write the text on the image
    cv2.putText(imgCenaOriginal, licPlaca.strCaracteres, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, ESCALA_AMARELO, intFontThickness)
# end function

###################################################################################################
if __name__ == "__main__":
    main()


















