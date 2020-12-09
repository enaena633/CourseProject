""" 
Coded by: Jack Ma, Ena Yu, and Matt McCarty, Team JEM University of Illinois at Urbana Champaign
CS 410 Text Information System Fall 2020 Final Project
"""
import numpy as np
import math
from datetime import datetime
import gensim as gn
from gensim import corpora
from gensim.parsing.preprocessing import preprocess_documents

def normalize(input_matrix):

    row_sums = input_matrix.sum(axis=1)
    try:
        assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    except Exception:
        raise Exception("Error while normalizing. Row(s) sum to zero")
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix

       
class plsa(object):

    def __init__(self, number_of_topics, iterations, mu):
        """
        Initialize a new plsa object with a new document, vocabulary, and matrices based on the number of topic and number of iterations given.
        """
        self.documents = []
        self.vocabulary = []
        self.term_doc_matrix = None 
        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z)
        self.topic_prob = None  # P(z | d, w)
        self.number_of_documents = []
        self.vocabulary_size = []
        self.number_of_topics = number_of_topics
        self.total_topics = 0
        self.mu = mu
        self.prior = None
        self.iterations = iterations

    def build_document(self):
        """ 
        Open up the pre-processed document, read it in, use the gensim preprocess_document function to process it (see gensim for documentation).
        Then build a vocabulary based on the processed documents.
        """
        with open('consolidated_nyt.tsv') as r:
            self.documents = r.read().splitlines()
        self.documents = preprocess_documents(self.documents)
        self.number_of_documents = len(self.documents)
        self.vocabulary = corpora.Dictionary(self.documents)
        self.vocabulary_size = len(self.vocabulary)
        print("Number of documents:" + str(len(self.documents)))
        print("Vocabulary size:" + str(self.vocabulary_size))

    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document, 
        and each column represents a vocabulary term.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """
        self.term_doc_matrix = np.zeros([self.number_of_documents, self.vocabulary_size], np.int16)
        for doc in range(self.number_of_documents):
            for word in self.vocabulary.doc2bow(self.documents[doc]):
                self.term_doc_matrix[doc, word[0]] = word[1]
            
    def EM_calc(self):
        # E-step updates P(z | w, d)        
        for j in range(self.total_topics): 
            self.topic_prob[j] = self.document_topic_prob[:, j, None] * self.topic_word_prob[None, j, :]
        denominators = np.sum(self.topic_prob, axis=0)
        denominators[denominators == 0] = 1
        self.topic_prob = self.topic_prob / denominators

        # M-step update P(z | d)
        for j in range(self.total_topics):
            self.document_topic_prob[:, j] = np.sum(np.multiply(self.term_doc_matrix, self.topic_prob[j]), axis=1)
        self.document_topic_prob /= np.sum(self.document_topic_prob, axis=1, keepdims=True) 
        
        # update P(w | z)
        for j in range(self.number_of_topics):
            self.topic_word_prob[j] = np.sum(np.multiply(self.term_doc_matrix, self.topic_prob[j]), axis=0)
            self.topic_word_prob[j] /= np.sum(self.topic_word_prob[j])
        for j in range(self.number_of_topics, self.total_topics):
            self.topic_word_prob[j] = np.sum(np.multiply(self.term_doc_matrix, self.topic_prob[j]), axis=0) + self.mu * self.prior[j-self.number_of_topics]
            self.topic_word_prob[j] /= (np.sum(self.topic_word_prob[j]) + self.mu)
    
    def calc_likelihood(self):
        loglikelihood = np.sum(np.multiply(self.term_doc_matrix, np.log(np.matmul(self.document_topic_prob, self.topic_word_prob))))
        if self.prior is None:
            return loglikelihood
        expanded_prior = np.zeros([self.total_topics, self.vocabulary_size])
        expanded_prior[self.number_of_topics:] = self.prior
        logMAP = loglikelihood + self.mu * np.sum((np.multiply(expanded_prior, np.log(self.topic_word_prob))))
        return logMAP

        """ loglikelihood = 0
        for i in range(0, self.number_of_documents):
            for j in range(0, self.vocabulary_size):
                tmp = 0
                for k in range(0, self.number_of_topics):
                    tmp += self.topic_word_prob[k, j] * self.document_topic_prob[i, k]
                if tmp > 0:
                    loglikelihood += self.term_doc_matrix[i, j] * math.log(tmp)        
        return loglikelihood """

    def initiate(self, prior):

        """
        Model topics.
        """
        print (datetime.now())

        self.total_topics = self.number_of_topics

        # determine if there is prior
        if prior is None:
            self.build_document()

        else:
            self.prior = prior
            self.total_topics += len(prior)
      
        # build term-doc matrix
        self.build_term_doc_matrix()
        
        # P(z | d, w)
        self.topic_prob = np.zeros([self.total_topics, self.number_of_documents, self.vocabulary_size], dtype=np.float)

        # P(z | d) 
        self.document_topic_prob = np.random.random(size = (self.number_of_documents, self.total_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        # P(w | z)
        self.topic_word_prob = np.random.random(size = (self.total_topics, self.vocabulary_size))
        if prior is not None:
            self.topic_word_prob.append(prior)
        self.topic_word_prob = normalize(self.topic_word_prob)

        # Run the EM algorithm
        for iteration in range(self.iterations):
            print("Iteration #" + str(iteration + 1) + "...")
            self.EM_calc()
            print(self.calc_likelihood())
            
        print(datetime.now())


def main():
    number_of_topics = 10
    max_iterations = 100
    mu = 30
    pl = plsa(number_of_topics, max_iterations, mu)    
    prior = None
    pl.initiate(prior)

if __name__ == '__main__':
    main()