# img_viewer.py
import csv
from cv2 import data
from pytesseract import Output
import numpy as np
import os
import cv2
from enum import auto
from pdf2image import convert_from_path
import PySimpleGUI as sg
import os.path
import matplotlib.pyplot as plt 
from operator import itemgetter
from decimal import Decimal
from re import sub
import pytesseract
import pandas as pd



per = 100
imgQ = cv2.imread('empresarial.jpg')
columnNames = []


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
#imgKp1 = cv2.drawKeypoints(imgQ,kp1,None)



months = ['ENE','FEB','MAR','ABR','MAY','JUN','JUL','AGO','SEP','OCT','NOV','DIC']
recibos = []





#Obtener variables unicas de arreglo
def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    
    return unique_list

#Funcion para poner label dentro de las graficas de barras
def autolabel(plot,label,ax):
    for idx,rect in enumerate(plot):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 0.5*height,
                label[idx],
                ha='center', va='bottom', rotation=90)

def informacion(empresas,recibos):
    return [
                [
                    sg.Text("Excel Generado de forma correcta",key='texto'), 
                ],
                [
                    sg.Text("Número de empresas: "+ str(empresas)), 
                ],
                [
                    sg.Text("Número de recibos: "+ str(recibos)), 
                ],
            ]

#Primera estructura de GUI
file_list_column = [
    [
        sg.Text("Seleccionar carpeta"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse()

    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
    [
        sg.Button("PROCESAR RECIBOS",key='procesar')
    ]
]


#Layout para iniciar GUI
layout = [
    [
        sg.Column(file_list_column,element_justification='center',key = 'principal'),
    ]
]
window = sg.Window("Recibos cfe", layout)


#Correr interfaz de GUI
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []
        fnames = [
            f
            for f in file_list
                if os.path.isfile(os.path.join(folder, f))
                and f.lower().endswith((".jpg",".png", ".pdf"))
                
        ]
        recibos = [
            f
            for f in file_list
                if os.path.isfile(os.path.join(folder, f))
                and f.lower().endswith((".jpg",".png", ".pdf"))
                
        ]
        
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            window["-TOUT-"].update(filename)
            window["-IMAGE-"].update(filename=filename)

        except:
            pass



    if event == "procesar":
        names = []
        rango = []
        series = []

        if(len(recibos)!=0):
            #create data array
            w = 22
            h = len(recibos)
            dataExcel = [[0 for x in range(w)] for y in range(h)] 
            dataConf = [[0 for x in range(18)] for y in range(h)] 
            dataConfInvert = [[0 for x in range(h)] for y in range(18)] 
            
            for i in range(len(recibos)):
                confidencia = 0
                img = cv2.imread(folder + '/'+recibos[i])
                columnNames.append(recibos[i])

                print(folder + '/'+recibos[i])

                #convierte pdf a jpg
                if recibos[i].lower().endswith((".pdf")):
                    pages = convert_from_path(folder + '/'+recibos[i])
                    count = 0
                    for page in pages:
                        if(count == 0):
                            page.save('output.jpeg', 'JPEG')
                        count +=1
                    img = cv2.imread('output.jpeg')
                
                # resize image
                width = 825
                height = 1275
                dsize = (width, height)
                img = cv2.resize(img, dsize)
                cv2.imshow(str(i)+"2",img)
                imgShow = img.copy()
                imgMask = np.zeros_like(imgShow)

                #Resetear variables
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

                
                #Confidence variables
                totalconf = 0
                servicioconf = 0
                rmuconf = 0
                periodoconf = 0
                tarifaconf = 0
                medidorconf = 0
                multiplicadorconf = 0
                cargaconf = 0
                contratadaconf = 0
                kWhBaseconf = 0
                kWhIntermedioconf = 0
                kWhPuntaconf = 0
                kWBaseconf = 0
                kWIntermedioconf = 0
                kWPuntaconf = 0
                KWMaxconf = 0
                kVArhconf = 0
                factorPotenciaconf = 0    

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

                    if(r[2] == 'text'):
                        if(r[3] ==' medidor'):
                            results = pytesseract.image_to_data(sepia_img, output_type=Output.DICT)
                        elif(r[3] ==' servicio' or r[3] ==' periodo' ):
                            results = pytesseract.image_to_data(rgb, output_type=Output.DICT) 
                        elif(r[3] == ' carga' or r[3] == ' kwbase'):
                            results = pytesseract.image_to_data(sepia_img, lang='eng',config='--psm 12 --oem 3 \ -c tessedit_char_whitelist=0123456789', output_type=Output.DICT)
                        else:
                            results = pytesseract.image_to_data(sepia_img, output_type=Output.DICT)
                        #  pytesseract.image_to_data(Image.open(tempFile), lang='eng', config='--psm 11', output_type=Output.DICT) 
                        
                        for j in range(0, len(results["text"])):
                            
                            if(r[3] ==' servicio'):
                                #Obtener SERBICIO
                                if(text == 'SERVICIO:'):
                                    servicio = results["text"][j]
                                    servicioconf = conf
                                #Obtener RMU
                                else:
                                    rmuconf = conf
                                    rmu+=str(results["text"][j] + ' ')
                                        
                                    
                            #Dato del peridoo
                            if(r[3] ==' periodo'):
                                periodoconf = conf
                                periodo +=str(results["text"][j])
                                
                            #Tabla datos
                            if(r[3] == ' tabla'):
                                ultimoDato = results["text"][j]
                                ultimoConf = results["conf"][j]

                                
                                if(text == 'intermedia' and arraryPositionIntermedio == 1):
                                    kWIntermedio = results["text"][j]
                                    kWIntermedioconf = conf
                                    arraryPositionIntermedio =  arraryPositionIntermedio+1
                                if(text == 'intermedia' and arraryPositionIntermedio == 0):
                                    kWhIntermedio = results["text"][j]
                                    kWhIntermedioconf = conf
                                    arraryPositionIntermedio =  arraryPositionIntermedio +1
                                if(text == 'punta' and arraryPositionPunta == 1):
                                    kWPunta= results["text"][j]
                                    kWPuntaconf = conf
                                    arraryPositionPunta =  arraryPositionPunta +1
                                if(text == 'punta' and arraryPositionPunta == 0):
                                    kWhPunta = results["text"][j]
                                    kWhPuntaconf = conf
                                    arraryPositionPunta = arraryPositionPunta +1
                                if(text == 'base' and arraryPositionBase == 1):
                                    confidencia = confidencia +conf
                                    kWBase= results["text"][j]
                                    kWBaseconf = conf
                                    arraryPositionBase =  arraryPositionBase +1
                                if(text == 'base' and arraryPositionBase == 0):
                                    kWhBase = results["text"][j]
                                    kWhBase = conf
                                    arraryPositionBase = arraryPositionBase +1
                                if(text =='KVArh' or text == 'kVArh' or text == 'KVAth'):
                                    kVArh = results["text"][j]
                                    kVArhconf = conf
                                if(KWMax5 =='KWM' or KWMax5 =='kWM' or text == 'kWMax'):
                                    confidencia = confidencia +conf
                                    KWMax = results["text"][j]
                                    KWMaxconf = conf


                            text = results["text"][j]
                            conf = int(results["conf"][j])
                            KWMax5 = text[:3]
                            texto = results["text"][j-1]

                        #Obtener TOTAL
                        if(r[3] ==' total'):
                            total = text
                            totalconf = conf
                            confidencia = confidencia +conf
                        elif(r[3] ==' medidor'):
                            medidor = text
                            medidorconf = conf
                        elif(r[3] ==' multiplicador'):
                            multiplicador = text
                            multiplicadorconf = conf
                        elif(r[3] ==' tarifa'):
                            tarifa = text
                            tarifaconf = conf
                        elif(r[3] ==' carga'):
                            carga = text
                            cargaconf = conf
                        elif(r[3] ==' contratada'):
                            contratada = text
                            contratadaconf = conf
                        elif(r[3] ==' kwbase'):
                            kWhBase = text
                            kWhBaseconf = conf
            
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

                dataExcel[i][0] = total
                dataExcel[i][1] = servicio
                dataExcel[i][2] = newRmu
                dataExcel[i][3] = periodoFecha
                dataExcel[i][4] = tarifa
                dataExcel[i][5] = medidor
                dataExcel[i][6] = multiplicador
                dataExcel[i][7] = carga
                dataExcel[i][8] = contratada
                dataExcel[i][9] = 'Na'
                dataExcel[i][10] = kWhIntermedio
                dataExcel[i][11] = kWhPunta
                dataExcel[i][12] = kWhBase
                dataExcel[i][13] = kWIntermedio
                dataExcel[i][14] = kWPunta
                dataExcel[i][15] = KWMax
                dataExcel[i][16] = kVArh
                dataExcel[i][17] = ultimoDato
                dataExcel[i][18] = 'Empresarial'

                dataConf[i][0] = totalconf
                dataConf[i][1] = servicioconf
                dataConf[i][2] = rmuconf
                dataConf[i][3] = periodoconf
                dataConf[i][4] = tarifaconf
                dataConf[i][5] = medidorconf
                dataConf[i][6] = multiplicadorconf
                dataConf[i][7] = cargaconf
                dataConf[i][8] = contratadaconf
                dataConf[i][9] = 0
                dataConf[i][10] = kWhIntermedioconf
                dataConf[i][11] = kWhPuntaconf
                dataConf[i][12] = kWhBaseconf
                dataConf[i][13] = kWIntermedioconf
                dataConf[i][14] = kWPuntaconf
                dataConf[i][15] = KWMaxconf
                dataConf[i][16] = kVArhconf
                dataConf[i][17] = ultimoConf

            
            for i in range(len(dataExcel)):
                graph_value = int(dataExcel[i][13])
                stringFecha = (dataExcel[i][3])
                value = Decimal(sub(r'[^\d.]', '', dataExcel[i][0]))
                dataExcel[i][21] = value
                series.append(dataExcel[i][1])

                for j in range(len(months)):
                    if(months[j] == stringFecha[2:5]):
                        dataExcel[i][19] = j+1
                dataExcel[i][20] = stringFecha[5:7]
            
            dataExcel= sorted(dataExcel, key=itemgetter(19))
            dataExcel= sorted(dataExcel, key=itemgetter(20))
            
            
            uniqueSeries = unique(series)
            uniqueSeries.reverse()

            for i in range(len(uniqueSeries)):
                names_p = []
                rango_p = []
                total_p = []
                label_p = []

                for j in range(len(dataExcel)):
                    stringFecha = (dataExcel[j][3])
                    firstMonth1 = (stringFecha[2:5])
                    firstYear1 = (stringFecha[5:7])
                    secondMonth1 =(stringFecha[10:13])
                    secondYear1 =(stringFecha[13:15])
                    if(uniqueSeries[i] == dataExcel[j][1]):
                        names_p.append(firstMonth1+firstYear1+'-'+secondMonth1+secondYear1)
                        rango_p.append(int(dataExcel[j][13]))
                        total_p.append(dataExcel[j][21])
                        label_p.append(dataExcel[j][0])
                
                f1 = plt.figure()
                ax1= f1.add_subplot(111)
                bar_plot = ax1.bar(names_p,rango_p)    
                autolabel(bar_plot,rango_p,ax1)
                ax1.set_title(' Grafica de consumo electrico | Serie: '+ uniqueSeries[i])
                ax1.tick_params(axis='x', which='major', labelsize=10,rotation = 45)

                f2 = plt.figure()
                ax2= f2.add_subplot(111)
                bar_plot2 = ax2.bar(names_p,total_p)
                autolabel(bar_plot2,label_p,ax2)    
                ax2.set_title(' Grafica de total | Serie: '+ uniqueSeries[i])
                ax2.tick_params(axis='x', which='major', labelsize=10,rotation = 45)

            window.extend_layout(window['principal'], informacion(len(uniqueSeries),len(dataExcel)))
            break



#Create csv with the data of the receipts
header = ['Total','Servicio','Rmu','Periodo','Tarifa','Medidor','Multiplicador',
'Carga','Contratada','kWhBase','kWhIntermedia','kWhPunta','kWBase','kWIntermedia','kWPunta','KWMax','KVArh','FactorPotenica','Tipo de recibo']
headerConf = ['Total','Servicio','Rmu','Periodo','Tarifa','Medidor','Multiplicador',
'Carga','Contratada','kWhBase','kWhIntermedia','kWhPunta','kWBase','kWIntermedia','kWPunta','KWMax','KVArh','FactorPotenica']

for i in range(18):
    for j in range(len(recibos)):
        dataConfInvert[i][j] = dataConf[j][i]


df = pd.DataFrame(dataConfInvert,columns = columnNames, index=headerConf)
df.plot.bar()
plt.title('Pytesseract Confidence')


with open('recibos_cfe.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(dataExcel)
    
window.close()    
plt.show()