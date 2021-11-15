# Kevin woolfolk a01251809 
from pytesseract import Output
import csv
import pytesseract
import numpy as np
import os
import cv2
import re
import PySimpleGUI as sg


per = 100
imgQ = cv2.imread('empresarial.jpg')
imgDomestic = cv2.imread('domestico.jpg')


roi =  [
        [(499, 139), (719, 196), 'text', ' total'], 
        [(38, 257), (477, 304), 'text', ' servicio'], 
        [(657, 261), (801, 301), 'text', ' periodo'],
        [(36, 307), (167, 356), 'text', ' tarifa'], 
        [(272, 319), (338, 357), 'text', ' medidor'], 
        [(468, 317), (498, 349), 'text', ' multiplicador'],
        [(214, 370), (246, 405), 'text', ' carga'],
        [(452, 461), (511, 488), 'text', ' kwbase'],
        [(466, 370), (496, 403), 'text', ' contratada'],
        [(40, 466), (502, 675), 'text', ' tabla']
    ]




orb = cv2.ORB_create(5000)
kp1, des1 = orb.detectAndCompute(imgQ,None)
kp3, des3 = orb.detectAndCompute(imgDomestic,None)
#imgKp1 = cv2.drawKeypoints(imgQ,kp1,None)

path = '/Users/kevinw/Documents/Tec de Monterrey/Ai/project/CFE-receipts-OCR/receipts'
myPicList = os.listdir(path)

w = 19
h = len(myPicList) - 2

dataExcel = [[0 for x in range(w)] for y in range(h)] 

for j,y in enumerate(myPicList):
    if(j>1):
        img = cv2.imread(path + "/" + y)
        
        kp2, des2 = orb.detectAndCompute(img,None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(des2,des1)
        matches.sort(key = lambda x: x.distance)
        good = matches[:int(len(matches)*(per/100))]
        imgMatch = cv2.drawMatches(img,kp2,imgQ,kp1,good[:20],None,flags=2)
        
        srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        M, _ = cv2.findHomography(srcPoints,dstPoints,cv2.RANSAC,5.0)
        imgScan = cv2.warpPerspective(img,M,(w,h))
        

        width = 825
        height = 1275

        # dsize
        dsize = (width, height)

        # resize image
        img = cv2.resize(img, dsize)
        cv2.imshow(y+"2",img)

        imgShow = img.copy()
        imgMask = np.zeros_like(imgShow)

        total = ''
        servicio = ''
        rmu = ''
        periodo=''
        tarifa = ''
        medidor = ''
        multiplicador =''
        carga = ''
        contratada =''

        kWhBase = ''
        kWhIntermedio = ''
        kWhPunta = ''
        kWBase = ''
        kWIntermedio = ''
        kWPunta = ''
        KWMax = ''
        kVArh = ''
        factorPotencia = ''        
        arraryPositionBase = 0
        arraryPositionIntermedio = 0
        arraryPositionPunta = 0

        for x,r in enumerate(roi):
            cv2.rectangle(imgMask,((r[0][0]),r[0][1]), ((r[1][0]),r[1][1]), (0,255,0),cv2.FILLED)
            imgShow = cv2.addWeighted(imgShow,0.99,imgMask,0.1,0)
            imgCrop = img[r[0][1]: r[1][1], r[0][0]: r[1][0]]
            img_grey = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
            th3 = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
            filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            # Applying cv2.filter2D function on our Logo image
            sharpen_img_2=cv2.filter2D(imgCrop,-1,filter)

            filter = np.array([[0.272, 0.534, 0.131],
                   [0.349, 0.686, 0.168],
                   [0.393, 0.769, 0.189]])
            # Applying cv2.transform function
            sepia_img=cv2.transform(imgCrop,filter)

            if(r[3] ==' tabla'):
                sepia_img = cv2.resize(sepia_img,(600,300))
            elif(r[3] ==' medidor' or r[3] == ' kwbase'):
                sepia_img = cv2.resize(sepia_img,(200,150))
            elif(r[3] ==' carga' or r[3] ==' periodo'  or r[3] ==' contratada' or r[3] ==' multiplicador' ):
                sepia_img = cv2.resize(sepia_img,(150,150))
                
            else:
                sepia_img = cv2.resize(sepia_img,(600,150))

            rgb = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2RGB)
            cv2.imshow(str(x),sepia_img)
    
            if(r[2]== 'text'):
                if(r[3] ==' medidor'):
                    results = pytesseract.image_to_data(sepia_img, output_type=Output.DICT)
                elif(r[3] ==' servicio' or r[3] ==' periodo' ):
                    results = pytesseract.image_to_data(rgb, output_type=Output.DICT) 
                elif(r[3] == ' carga' or r[3] == ' kwbase'):
                    results = pytesseract.image_to_data(sepia_img, lang='eng',config='--psm 12 --oem 3 \ -c tessedit_char_whitelist=0123456789', output_type=Output.DICT)
                else:
                    results = pytesseract.image_to_data(sepia_img, output_type=Output.DICT)
                #  pytesseract.image_to_data(Image.open(tempFile), lang='eng', config='--psm 11', output_type=Output.DICT) 
                
                for i in range(0, len(results["text"])):
                    
                    if(r[3] ==' servicio'):
                        #Obtener SERBICIO
                        if(text == 'SERVICIO:'):
                            servicio = results["text"][i]
                        #Obtener RMU
                        else:
                            rmu+=str(results["text"][i] + ' ')
                                 
                            
                    #Dato del peridoo
                    if(r[3] ==' periodo'):
                         periodo +=str(results["text"][i])
                         
                    #Tabla datos
                    if(r[3] == ' tabla'):
                        ultimoDato = results["text"][i]

                        
                        if(text == 'intermedia' and arraryPositionIntermedio == 1):
                            kWIntermedio = results["text"][i]
                            arraryPositionIntermedio =  arraryPositionIntermedio+1
                        if(text == 'intermedia' and arraryPositionIntermedio == 0):
                            kWhIntermedio = results["text"][i]
                            arraryPositionIntermedio =  arraryPositionIntermedio +1
                        if(text == 'punta' and arraryPositionPunta == 1):
                            kWPunta= results["text"][i]
                            arraryPositionPunta =  arraryPositionPunta +1
                        if(text == 'punta' and arraryPositionPunta == 0):
                            kWhPunta = results["text"][i]
                            arraryPositionPunta = arraryPositionPunta +1
                        if(text == 'base' and arraryPositionBase == 1):
                            kWBase= results["text"][i]
                            arraryPositionBase =  arraryPositionBase +1
                        if(text == 'base' and arraryPositionBase == 0):
                            kWhBase = results["text"][i]
                            arraryPositionBase = arraryPositionBase +1
                        if(text =='KVArh' or text == 'kVArh' or text == 'KVAth'):
                            kVArh = results["text"][i]
                        if(KWMax5 =='KWM' or KWMax5 =='kWM' or text == 'kWMax'):
                            KWMax = results["text"][i]


                    text = results["text"][i]
                    KWMax5 = text[:3]
                    texto = results["text"][i-1]

                        
                    
                    
                    
                #print(r[3])
                #Obtener TOTAL
                if(r[3] ==' total'):
                    total = text
                elif(r[3] ==' medidor'):
                    medidor = text
                elif(r[3] ==' multiplicador'):
                    multiplicador = text
                elif(r[3] ==' tarifa'):
                    tarifa = text
                elif(r[3] ==' carga'):
                    carga = text
                elif(r[3] ==' contratada'):
                    contratada = text
                elif(r[3] ==' kwbase'):
                    kWhBase = text
                    print(kWhBase)


        periodoFecha = ''
        periodoArray = []
        periodoArray = str.split(periodo)
        if(len(periodoArray)>1):
            periodoFecha = periodoArray[1]
        else:
            periodoFecha = periodoArray[0]

        rmu = list(rmu)
        newRmu = ''
        cont = 0
        rmuReady = False
        for k in range(len(rmu)):
            if(rmu[k].isdigit()):
                rmuReady = True
            if(rmuReady):
                newRmu += str(rmu[k])

        if(kVArh == ultimoDato):
            ultimoDato = 'No Existe'

        dataExcel[j-2][0] = total
        dataExcel[j-2][1] = servicio
        dataExcel[j-2][2] = newRmu
        dataExcel[j-2][3] = periodoFecha
        dataExcel[j-2][4] = tarifa
        dataExcel[j-2][5] = medidor
        dataExcel[j-2][6] = multiplicador
        dataExcel[j-2][7] = carga
        dataExcel[j-2][8] = contratada
        dataExcel[j-2][9] = 'Na'
        dataExcel[j-2][10] = kWhIntermedio
        dataExcel[j-2][11] = kWhPunta
        dataExcel[j-2][12] = kWhBase
        dataExcel[j-2][13] = kWIntermedio
        dataExcel[j-2][14] = kWPunta
        dataExcel[j-2][15] = KWMax
        dataExcel[j-2][16] = kVArh
        dataExcel[j-2][17] = ultimoDato
        dataExcel[j-2][18] = 'Empresarial'

        cv2.imshow(y +"2",imgShow)



#Create csv with the data of the receipts
header = ['Total','Servicio','Rmu','Periodo','Tarifa','Medidor','Multiplicador',
'Carga','Contratada','kWhBase','kWhIntermedia','kWhPunta','kWBase','kWIntermedia','kWPunta','KWMax','KVArh','FactorPotenica','Tipo de recibo']

with open('cfe_info.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(dataExcel)


print(dataExcel)
print('Done')
#cv2.imshow('KeyPoints',imgKp1)
#cv2.imshow('Output',imgQ)
cv2.waitKey(0)

