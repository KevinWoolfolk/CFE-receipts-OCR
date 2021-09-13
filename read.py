# Kevin woolfolk a01251809 
from pytesseract import Output
import csv
import pytesseract
import cv2

#initial variables
receipts =['receipts/prueba1.jpg','receipts/prueba2.jpg','receipts/prueba3.jpg','receipts/prueba4.jpg','receipts/prueba5.jpg','receipts/prueba6.jpg']
w = 4
h = len(receipts)
data = [[0 for x in range(w)] for y in range(h)] 

# loop over each receipt
for j in range(0, len(receipts)):

    #initialize values
    total ='$0.00'
    firstmonth = 'NONE'
    secondmonth = 'NONE'
    energy = '0'
    whichmonth = 0
    months = ['ENE','FEB','MAR','ABR','MAY','JUN','JUL','AGO','SEP','OCT','NOV','DIC']
    facturado = False
    estimada = False
    energyOut = 0

    #ocr the receipt
    image = cv2.imread(receipts[j])
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pytesseract.image_to_data(rgb, output_type=Output.DICT)

    # loop over each of the individual text localizations
    for i in range(0, len(results["text"])):
       
        #Traduction of the text
        text = results["text"][i]

         #Confidence of the text
        conf = int(results["conf"][i])
        
        # filter out weak confidence text localizations, and text that have len 0 and 1
        if (conf > 50 and format(len(text)) !='0' and  format(len(text))!='1'):

            word = format(text)

            #Get the total number of the receipt
            if(format(results["text"][i-1]) == 'Total'):
                total = word

            #Set true facturado
            if(word == 'FACTURADO:'):
                facturado = True

            #Set true estimado
            if(word == 'Estimada'):
                estimada = True

            #Get second month
            if(word in months and whichmonth == 1 and len(word)==3 and facturado):
                secondmonth = word

            #Get first month
            if(word in months and whichmonth == 0 and len(word)==3 and facturado):
                firstmonth = word
                whichmonth = whichmonth+1

            #Get energy
            if(len(results["text"][i-1]) == 5  and  estimada and energyOut==0):
                energy = word
                energyOut =energyOut+1
                
        
    #fill the array 
    data[j][0] = total
    data[j][1] = energy
    data[j][2] = firstmonth
    data[j][3] = secondmonth
    

#Create csv with the data of the receipts
header = ['Total', 'Energy', 'First month period', 'Second month period']
with open('receipts_info.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(data)
