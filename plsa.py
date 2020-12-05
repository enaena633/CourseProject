import numpy as np
import math
from datetime import datetime
from dateutil import parser
import gensim as gn
from gensim import corpora
from gensim.parsing.preprocessing import remove_stopwords, strip_numeric, strip_non_alphanum, strip_short

def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    row_sums = input_matrix.sum(axis=1)
    try:
        assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    except Exception:
        raise Exception("Error while normalizing. Row(s) sum to zero")
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix

       
class Corpus(object):

    """
    A collection of documents.
    """

    def __init__(self, documents_path):
        """
        Initialize empty document list.
        """
        self.documents = []
        self.vocabulary = None
        self.likelihoods = []
        self.date_to_document = {}
        self.documents_path = documents_path
        self.term_doc_matrix = None 
        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z)
        self.topic_prob = None  # P(z | d, w)

        self.number_of_documents = 0
        self.vocabulary_size = 0

    def build_corpus(self):
        """
        Read document, fill in self.documents, a list of list of word
        self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]
        
        Update self.number_of_documents
        """
        docs = []
        f = open(self.documents_path, "r")
        for line in f:
            docs.append(line.lower())
            self.number_of_documents += 1
        f.close()

        for i, doc in enumerate(docs):
            doc_date = doc.split("\t")[0]
            if doc_date in self.date_to_document:
                self.date_to_document[doc_date].append(i)
            else:
                self.date_to_document[doc_date] = [i]

            cleaned_doc = strip_short(remove_stopwords(strip_numeric(strip_non_alphanum(doc))), minsize=3)
            self.documents.append(cleaned_doc)

    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus. Put it in self.vocabulary
        for example: ["rain", "the", ...]

        Update self.vocabulary_size
        """      
        texts = [[text for text in doc.split()] for doc in self.documents]
        self.vocabulary = corpora.Dictionary(texts)
        self.vocabulary_size = len(self.vocabulary.keys())

    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document, 
        and each column represents a vocabulary term.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """
        self.term_doc_matrix = np.zeros([self.number_of_documents, len(self.vocabulary.keys())], np.int8)
        for line in range(self.number_of_documents):
            for word in self.documents[line]:
                self.term_doc_matrix[line][self.vocabulary.get(word)] += 1
        

    def initialize_randomly(self, number_of_topics):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob

        Don't forget to normalize! 
        HINT: you will find numpy's random matrix useful [https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.random.html]
        """
        self.document_topic_prob = np.random.random(size = (self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.random.random(size = (number_of_topics, self.vocabulary_size))
        self.topic_word_prob = normalize(self.topic_word_prob) 
        

    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform 
        probability distribution. This is used for testing purposes.

        DO NOT CHANGE THIS FUNCTION
        """
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.ones((number_of_topics, self.vocabulary_size))
        self.topic_word_prob = normalize(self.topic_word_prob)

    def initialize(self, number_of_topics, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        print("Initializing...")

        if random:
            self.initialize_randomly(number_of_topics)
        else:
            self.initialize_uniformly(number_of_topics)

    def expectation_step(self, number_of_topics):
        """ The E-step updates P(z | w, d)
        """

        for j in range(number_of_topics): 
            self.topic_prob[j] = np.expand_dims(self.document_topic_prob[:, j], 1) * np.expand_dims(self.topic_word_prob[j, :], 0)
        self.topic_prob = self.topic_prob / np.sum(self.topic_prob, axis=0)

    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z)
        """     
        # update P(w | z)

        for j in range(number_of_topics):
            self.topic_word_prob[j] = np.sum(np.multiply(self.term_doc_matrix, self.topic_prob[j]), axis=0)
            self.topic_word_prob[j] /= np.sum(self.topic_word_prob[j])
        
        # update P(z | d)
        for j in range(number_of_topics):
            self.document_topic_prob[:, j] = np.sum(np.multiply(self.term_doc_matrix, self.topic_prob[j]), axis=1)
            self.document_topic_prob[:, j] /= np.sum(self.document_topic_prob[:, j]) 

    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        
        Append the calculated log-likelihood to self.likelihoods

        """                

    def plsa(self, number_of_topics, max_iter, epsilon, mu, prior):

        """
        Model topics.
        """
        print (datetime.now())
        
        # build term-doc matrix
        self.build_term_doc_matrix()
        
        # Create the counter arrays.
        
        # P(z | d, w)
        self.topic_prob = np.zeros([number_of_topics, self.number_of_documents, self.vocabulary_size], dtype=np.float)

        # P(z | d) P(w | z)
        self.initialize(number_of_topics, random=True)

        # Run the EM algorithm
        current_likelihood = 0.0

        for iteration in range(max_iter):
            print("Iteration #" + str(iteration + 1) + "...")
            self.expectation_step(number_of_topics)
            self.maximization_step(number_of_topics)
            
        print(datetime.now())


def main():
    documents_path = 'sanitized_nyt.tsv'
    corpus = Corpus(documents_path)  # instantiate corpus
    corpus.build_corpus()
    corpus.build_vocabulary()
    print(corpus.vocabulary)
    print("Vocabulary size:" + str(corpus.vocabulary_size))
    print("Number of documents:" + str(len(corpus.documents)))
    number_of_topics = 2
    max_iterations = 50
    epsilon = 0.001
    mu = 30
    prior = None
    corpus.plsa(number_of_topics, max_iterations, epsilon, mu, prior)


if __name__ == '__main__':
    main()
