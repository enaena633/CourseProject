import gensim as gn
from gensim import corpora
from gensim.parsing.preprocessing import remove_stopwords, strip_numeric, strip_non_alphanum, strip_short

def main():
    documents_path = 'sanitized_nyt.tsv'
    documents = []
    cleaned_docs =[]
    texts = []
    number_of_documents = 0
    f = open(documents_path, "r")
    for line in f:
        documents.append(line.lower())
        number_of_documents += 1
    f.close()
    
    for document in documents:
        cleaned_doc = strip_short(remove_stopwords(strip_numeric(strip_non_alphanum(document))), minsize=3)
        cleaned_docs.append(cleaned_doc)

    texts = [[text for text in doc.split()] for doc in cleaned_docs]
    dictionary = corpora.Dictionary(texts)
    print(number_of_documents)
    print(dictionary)

if __name__ == '__main__':
    main()