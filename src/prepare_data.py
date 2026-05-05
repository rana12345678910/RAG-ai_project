from PyPDF2 import PdfReader

#data loading
def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

doc = load_pdf("corpus_juridique/Code civil.pdf")
print(doc[:500])  

#chunks
def chunk_text(text, size=500):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

chunks = chunk_text(doc)
