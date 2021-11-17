from pdf2image import convert_from_path
pages = convert_from_path('prueba.pdf')
count = 0
for page in pages:
    if(count == 0):
        page.save('output.jpeg', 'JPEG')
    count +=1