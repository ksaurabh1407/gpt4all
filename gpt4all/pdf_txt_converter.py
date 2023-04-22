# importing required modules
import PyPDF2

# create a pdf file object
pdfFileObj = open('./data/RFP.pdf', 'rb')

# create a pdf reader object
pdfReader = PyPDF2.PdfReader(pdfFileObj)

# print number of pages in the pdf file
print("Page Number:", len(pdfReader.pages))

for x in range(len(pdfReader.pages)):
  # create a page object
  pageObj = pdfReader.pages[x]

  # extract text from page
  text = pageObj.extract_text()

  # display just the text
  print(text)

  # save to a text file for later use
  # copy the path where the script and pdf is placed
  file1=open(r"./data/RFP.txt","a")
  file1.writelines(text)

# closing the pdf file object
pdfFileObj.close()

# closing the text file object
file1.close()