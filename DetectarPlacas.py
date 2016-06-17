# DetectarPlacas.py
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math
import Main
import random

import Preprocesso
import DetectarCaracteres
import PossivelPlaca
import PossivelCaractere

# variáveis de nível de módulo ##########################################################################
PLACA_LARGURA_FATOR_PREENCHIMENTO = 1.3
PLACA_ALTURA_FATOR_PREENCHIMENTO = 1.5

###################################################################################################
def DetectarPlacasInScene(imgCenaOriginal):
    listaDePossiveisPlacas = []                   # este será o valor de retorno

    altura, largura, numCanais = imgCenaOriginal.shape

    imgEscalaDeCinzaScene = np.zeros((altura, largura, 1), np.uint8)
    imgThresholdScene = np.zeros((altura, largura, 1), np.uint8)
    imgContours = np.zeros((altura, largura, 3), np.uint8)

    cv2.destroyAllWindows()

    if Main.mostrarPassos == True: # Mostrar etapas #######################################################
        cv2.imshow("0", imgCenaOriginal)
    # end if # Mostrar etapas #########################################################################

    imgEscalaDeCinzaScene, imgThresholdScene = Preprocesso.Preprocesso(imgCenaOriginal)
    # Preproces(franklin que quiz botar assim)(ou preprocessamento) para obter imagens em tons de cinza e limiar

    if Main.mostrarPassos == True: # Mostrar etapas #######################################################
        cv2.imshow("1a", imgEscalaDeCinzaScene)
        cv2.imshow("1b", imgThresholdScene)
    # end if # Mostrar etapas #########################################################################

        # Encontrar todos os caracteres possíveis na cena,
        # Esta função primeira encontra todos os Contornos, então só inclui Contornos
        # Que poderia ser caracteres (sem comparação com outros caracteres até o momento)
    listaDePossiveisCaracteresInScene = findPossivelCaracteresInScene(imgThresholdScene)

    if Main.mostrarPassos == True: # Mostrar etapas #######################################################
        print ("step 2 - len(listaDePossiveisCaracteresInScene) = " + str(len(listaDePossiveisCaracteresInScene)))         # 131 with MCLRNF1 image

        imgContours = np.zeros((altura, largura, 3), np.uint8)

        contornos = []

        for possivelCaractere in listaDePossiveisCaracteresInScene:
            contornos.append(possivelCaractere.contour)
        # end for

        cv2.drawContours(imgContours, contornos, -1, Main.ESCALA_BRANCO)
        cv2.imshow("2b", imgContours)
    # end if # Mostrar etapas #########################################################################

        # Dada uma lista de todos os caracteres possíveis, encontrar grupos de caracteres correspondentes
        # Nas próximas etapas cada grupo de caracteres correspondentes tentará ser reconhecida como uma Placa
    listaDeListasDeCombinacaoDeCaracteresInScene = DetectarCaracteres.findListOfListsOfMatchingCaracteres(listaDePossiveisCaracteresInScene)

    if Main.mostrarPassos == True: # Mostrar etapas #######################################################
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
    # end if # Mostrar etapas #########################################################################

    for listaDeCombinacaoDeCaracteres in listaDeListasDeCombinacaoDeCaracteresInScene:       # Para cada grupo de gráficos correspondentes
        possivelPlaca = extrairPlaca(imgCenaOriginal, listaDeCombinacaoDeCaracteres)         # tentar extrair Placa

        if possivelPlaca.imgPlaca is not None:                          # if Placa foi encontrado
            listaDePossiveisPlacas.append(possivelPlaca)                  # adicionar à lista de possíveis placas
        # end if
    # end for

    print ("\n" + str(len(listaDePossiveisPlacas)) + " possíveis placas encontrados")

    if Main.mostrarPassos == True: # Mostrar etapas #######################################################
        print ("\n")
        cv2.imshow("4a", imgContours)

        for i in range(0, len(listaDePossiveisPlacas)):
            p2fRectPoints = cv2.boxPoints(listaDePossiveisPlacas[i].rrLocationOfPlacaInScene)

            cv2.line(imgContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), Main.ESCALA_VERMELHO, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), Main.ESCALA_VERMELHO, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), Main.ESCALA_VERMELHO, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), Main.ESCALA_VERMELHO, 2)

            cv2.imshow("4a", imgContours)

            print ("possível Placa " + str(i) + ", clique em qualquer imagem e pressione uma tecla para continuar. . .")

            cv2.imshow("4b", listaDePossiveisPlacas[i].imgPlaca)
            cv2.waitKey(0)
        # end for

        print ("\ndetecção de Placa completa, clique em qualquer imagem e pressione uma tecla para iniciar o reconhecimento de caractere . . .\n")
        cv2.waitKey(0)
    # end if # Mostrar etapas #########################################################################

    return listaDePossiveisPlacas
# end function

###################################################################################################
def findPossivelCaracteresInScene(imgThreshold):
    listaDePossiveisCaracteres = []                # este será o valor de retorno

    intCountOfPossivelCaracteres = 0

    imgThresholdCopia = imgThreshold.copy()

    imgContours, contornos, npaHierarchy = cv2.findContours(imgThresholdCopia, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   # encontrar todos os Contornos

    altura, largura = imgThreshold.shape
    imgContours = np.zeros((altura, largura, 3), np.uint8)

    for i in range(0, len(contornos)):                       # para cada contorno

        if Main.mostrarPassos == True: # Mostrar etapas ###################################################
            cv2.drawContours(imgContours, contornos, i, Main.ESCALA_BRANCO)
        # end if # Mostrar etapas #####################################################################

        possivelCaractere = PossivelCaractere.PossivelCaractere(contornos[i])

        if DetectarCaracteres.verificaSePossivelCaractere(possivelCaractere):         # Se o contorno é uma possível char, note que este não se compara a outros caracteres (até o momento) . . .
            intCountOfPossivelCaracteres = intCountOfPossivelCaracteres + 1           # contagem de incremento de caracteres possíveis
            listaDePossiveisCaracteres.append(possivelCaractere)                      # e adicionar à lista de possíveis caracteres
        # end if
    # end for

    if Main.mostrarPassos == True: # Mostrar etapas #######################################################
        print ("\netapa 2 - len(contornos) = " + str(len(contornos)))
        print ("etapa 2 - intCountOfPossivelCaracteres = " + str(intCountOfPossivelCaracteres))
        cv2.imshow("2a", imgContours)
    # end if # Mostrar etapas #########################################################################

    return listaDePossiveisCaracteres
# end function


###################################################################################################
def extrairPlaca(imgOriginal, listaDeCombinacaoDeCaracteres):
    possivelPlaca = PossivelPlaca.PossivelPlaca()           # este será o valor de retorno

    listaDeCombinacaoDeCaracteres.sort(key = lambda matchingCaractere: matchingCaractere.intCenterX)        # tipo caracteres da esquerda para a direita com base na posição x

            # calcular o ponto central da Placa
    fltPlacaCenterX = (listaDeCombinacaoDeCaracteres[0].intCenterX + listaDeCombinacaoDeCaracteres[len(listaDeCombinacaoDeCaracteres) - 1].intCenterX) / 2.0
    fltPlacaCenterY = (listaDeCombinacaoDeCaracteres[0].intCenterY + listaDeCombinacaoDeCaracteres[len(listaDeCombinacaoDeCaracteres) - 1].intCenterY) / 2.0

    ptPlacaCenter = fltPlacaCenterX, fltPlacaCenterY

            # calcular Largura e altura da Placa
    intPlacaWidth = int((listaDeCombinacaoDeCaracteres[len(listaDeCombinacaoDeCaracteres) - 1].intBoundingRectX + listaDeCombinacaoDeCaracteres[len(listaDeCombinacaoDeCaracteres) - 1].intBoundingRectWidth - listaDeCombinacaoDeCaracteres[0].intBoundingRectX) * PLACA_LARGURA_FATOR_PREENCHIMENTO)

    intTotalOfCaractereHeights = 0

    for matchingCaractere in listaDeCombinacaoDeCaracteres:
        intTotalOfCaractereHeights = intTotalOfCaractereHeights + matchingCaractere.intBoundingRectHeight
    # end for

    fltAverageCaractereHeight = intTotalOfCaractereHeights / len(listaDeCombinacaoDeCaracteres)

    intPlacaHeight = int(fltAverageCaractereHeight * PLACA_ALTURA_FATOR_PREENCHIMENTO)

            # calcular o ângulo de correção da região Placa
    fltOpposite = listaDeCombinacaoDeCaracteres[len(listaDeCombinacaoDeCaracteres) - 1].intCenterY - listaDeCombinacaoDeCaracteres[0].intCenterY
    fltHypotenuse = DetectarCaracteres.distanciaEntreCaracteres(listaDeCombinacaoDeCaracteres[0], listaDeCombinacaoDeCaracteres[len(listaDeCombinacaoDeCaracteres) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)
            # ponto central da região, Largura e altura e ângulo de correção da variável em rotação de retângulo de Placa
    possivelPlaca.rrLocationOfPlacaInScene = ( tuple(ptPlacaCenter), (intPlacaWidth, intPlacaHeight), fltCorrectionAngleInDeg )

            # passos finais são para realizar a rotação real

            # obter a matriz de rotação para o nosso ângulo de correção calculado
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlacaCenter), fltCorrectionAngleInDeg, 1.0)

    altura, largura, numCanais = imgOriginal.shape      # descompactar imagem original Largura e altura

    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (largura, altura))       # girar a imagem inteira

    imgCropped = cv2.getRectSubPix(imgRotated, (intPlacaWidth, intPlacaHeight), tuple(ptPlacaCenter))

    possivelPlaca.imgPlaca = imgCropped         # copiar a imagem Placa cortada na variável membro aplicável à possível Placa

    return possivelPlaca
# end function























