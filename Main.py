# Main.py
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os

import DetectarCaracteres
import DetectarPlacas
import PossivelPlaca

# variáveis de nível de módulo ##########################################################################
ESCALA_PRETO = (0.0, 0.0, 0.0)
ESCALA_BRANCO = (255.0, 255.0, 255.0)
ESCALA_AMARELO = (0.0, 255.0, 255.0)
ESCALA_VERDE = (0.0, 255.0, 0.0)
ESCALA_VERMELHO = (0.0, 0.0, 255.0)

mostrarPassos = False


###################################################################################################
def main():
    blnKNNTrainingSuccessful = DetectarCaracteres.loadKNNDataAndTrainKNN()
    # tentativa KNN(o vizinho mais proximo, algoritmo) training

    if blnKNNTrainingSuccessful == False:
        # se KNN(o vizinho mais proximo, algoritmo) training não foi bem sucedida
        print ("\nerror: KNN traning was not successful\n")
        # mostrar mensagem de erro
        return
        # e fechar programa
    # end if

    imgCenaOriginal = cv2.imread("imagens/f15.jpg")
    # abrir imagem

    if imgCenaOriginal is None:
        # se a imagem não foi lida com sucesso
        print ("\nErro: Arquivo de imagem não lido\n\n")
        # exibir mensagem de erro de impressão
        os.system("pause")
        # pausar assim que o usuário puder ver a mensagem de erro
        return
        # e fechar programa
    # end if

    listaDePossiveisPlacas = DetectarPlacas.DetectarPlacasInScene(imgCenaOriginal)
    # detectar Placas

    listaDePossiveisPlacas = DetectarCaracteres.DetectarCaracteresNasPlacas(listaDePossiveisPlacas)
    # detectar caracteres nas Placas

    cv2.imshow("imgCenaOriginal", imgCenaOriginal)
    # exibir imagem que foi escolhida cv2.imread("imagens/f15.jpg")

    if len(listaDePossiveisPlacas) == 0:
        # se não foram encontradas Placas
        print ("\nNenhuma placa foi encontrada\n")
        # informar ao usuário que a placa não foi encontrada
    else:
        # else

        # Se entrar aqui lista de possíveis Placas tem pelo menos uma Placa

        # Classificar a lista de possíveis Placas em ordem decrescente (maior número de caracteres para o menos número de caracteres)
        listaDePossiveisPlacas.sort(key=lambda possivelPlaca: len(possivelPlaca.strCaracteres), reverse=True)

        # Suponha que o local com os caracteres mais reconhecidos é a placa real
        licPlaca = listaDePossiveisPlacas[0]

        cv2.imshow("imgPlaca", licPlaca.imgPlaca)
        # mostrar corte do lugar e limite de Placa
        cv2.imshow("imgThreshold", licPlaca.imgThreshold)

        if len(licPlaca.strCaracteres) == 0:
            # Se nenhum caractere foi encontrado na Placa
            print ("\nNenhum caractere foi encontrado\n\n")
            # mostrar mensagem
            return
            # e fechar programa
        # end if

        desenharRetanguloVermelhoAoRedorDaPlaca(imgCenaOriginal, licPlaca)
        # desenhar um retângulo vermelho em torno de Placa

        print ("\nPlaca lida da imagem = " + licPlaca.strCaracteres + "\n")
        # escrever o texto da placa para std out
        print ("----------------------------------------")

        escreverCaracteresDaPlacaNaImagem(imgCenaOriginal, licPlaca)
        # escrever o texto da placa na imagem

        cv2.imshow("imgCenaOriginal", imgCenaOriginal)
        #exibir de novo a imagem, só que alterada. com as inserções

        cv2.imwrite("imgCenaOriginal.png", imgCenaOriginal)
        # gravar essa imagem alterada para o arquivo

    # end if else

    cv2.waitKey(0)
    # mantenha as janelas abertas até que o usuário pressiona uma tecla

    return


# end main

###################################################################################################
def desenharRetanguloVermelhoAoRedorDaPlaca(imgCenaOriginal, licPlaca):
    p2fRectPoints = cv2.boxPoints(licPlaca.rrLocationOfPlacaInScene)
    # obter 4 vértices do retângulo girado

    cv2.line(imgCenaOriginal, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), ESCALA_VERMELHO, 2)
    # desenhar 4 linhas vermelhas
    cv2.line(imgCenaOriginal, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), ESCALA_VERMELHO, 2)
    cv2.line(imgCenaOriginal, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), ESCALA_VERMELHO, 2)
    cv2.line(imgCenaOriginal, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), ESCALA_VERMELHO, 2)


# end function

###################################################################################################
def escreverCaracteresDaPlacaNaImagem(imgCenaOriginal, licPlaca):
    ptCenterOfTextAreaX = 0
    # este será o centro da área o texto será escrito para
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0
    # este será o canto inferior esquerdo da área que o texto será escrito para
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgCenaOriginal.shape
    PlacaHeight, PlacaWidth, PlacaNumChannels = licPlaca.imgPlaca.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX
    # escolher uma fonte para os caracteres
    fltFontScale = float(PlacaHeight) / 30.0
    # escala fonte base na altura da área da placa
    intFontThickness = int(round(fltFontScale * 1.5))
    # espessura da fonte base na escala de fonte

    textSize, baseline = cv2.getTextSize(licPlaca.strCaracteres, intFontFace, fltFontScale, intFontThickness)
    # chamar setTextSize

    # descompactar retângulo girado em ponto central, Largura e altura e ângulo
    ((intPlacaCenterX, intPlacaCenterY), (intPlacaWidth, intPlacaHeight),
     fltCorrectionAngleInDeg) = licPlaca.rrLocationOfPlacaInScene

    intPlacaCenterX = int(intPlacaCenterX)
    # certifique-se o centro é um inteiro
    intPlacaCenterY = int(intPlacaCenterY)

    ptCenterOfTextAreaX = int(intPlacaCenterX)
    # a localização horizontal da área de texto é o mesmo que a Placa

    if intPlacaCenterY < (sceneHeight * 0.75):
        # se a placa é na parte superior 3/4(tambem conhecido como 0,75) da imagem
        ptCenterOfTextAreaY = int(round(intPlacaCenterY)) + int(round(PlacaHeight * 1.6))
        # escrever os caracteres em baixo da Placa
    else:
        # senão se a placa é na parte inferior 1/4(tambem conhecido como 0,25) da imagem
        ptCenterOfTextAreaY = int(round(intPlacaCenterY)) - int(round(PlacaHeight * 1.6))
        # escrever os caracteres em cima da Placa
    # end if

    textSizeWidth, textSizeHeight = textSize
    # tamanho do texto descompactar Largura e altura

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))
    # calcular a origem inferior esquerda da área de texto
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))
    # com base no centro textarea, Largura, e Altura

    # escreva o texto na imagem
    cv2.putText(imgCenaOriginal, licPlaca.strCaracteres, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace,
                fltFontScale, ESCALA_AMARELO, intFontThickness)


# end function

###################################################################################################
if __name__ == "__main__":
    main()



















