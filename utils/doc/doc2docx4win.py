import os
from win32com import client as wc

def doc2docx(doc_path, docx_path):
    word = wc.Dispatch("Word.Application")
    doc = word.Documents.Open(doc_path)
    doc.SaveAs(docx_path, 16)
    doc.Close()
    word.Quit()

def BathDoc2Docx(doc_dir, docx_dir):
    if not os.path.exists(docx_dir):
        os.makedirs(docx_dir)
    for i, folder in enumerate(os.listdir(doc_dir)):
        print("=========== Deal with the folder %s[%s th] ==========\n" %(folder, i))
        doc_path = os.path.join(doc_dir, folder)
        docx_path = os.path.join(docx_dir, folder + 'x')
        doc2docx(doc_path, docx_path)

def main():
    doc_root = 'C:/Users/xulingxiao.EM/Documents/hena/2018'
    docx_root = 'C:/Users/xulingxiao.EM/Documents/hena/2018_docx'
    for item in os.listdir(doc_root):
        doc_dir = os.path.join(doc_root, item)
        docx_dir = os.path.join(docx_root, item)
        BathDoc2Docx(doc_dir, docx_dir)

if __name__ == '__main__':
    main()
