""" 
Coded by: Jack Ma, Ena Yu, and Matt McCarty, Team JEM University of Illinois at Urbana Champaign
CS 410 Text Information System Fall 2020 Final Project
"""
import numpy as np
import gensim as gn
from gensim import corpora
from gensim.parsing.preprocessing import preprocess_documents

def normalize(input_matrix):
    # This method was copied directly from MP3 with no changes, all credit to the writer of MP3
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
        Initialize a new plsa object with a empty document, vocabulary, and matrices based on the number of topic, number of iterations, and mu.
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
        self.total_topics = number_of_topics
        self.mu = mu
        self.prior = None
        self.iterations = iterations

    def __build_document(self):
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

    def __build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document, and each column represents a vocabulary term.
        This is accomplished by using doc2bow (see gensim) method on the vocabulary dictionary. 
        Checking each document in documents for the number of occurances of a word in the vocabulary. Then putting that count of occurances into the term_doc_matrix.
        """
        self.term_doc_matrix = np.zeros([self.number_of_documents, self.vocabulary_size], np.int16)
        for doc in range(self.number_of_documents):
            for word in self.vocabulary.doc2bow(self.documents[doc]):
                self.term_doc_matrix[doc, word[0]] = word[1]
            
    def __EM_calc(self):
        """
        This is where the magic of PLSA happens. This uses numpy matrix manipulation to compute the EM steps in PLSA.
        The second part of the M step take prior into consideration if there is a prior.
        """
        # E-step updates P(z | w, d)        
        for j in range(self.total_topics): 
            self.topic_prob[j] = self.document_topic_prob[:, j, None] * self.topic_word_prob[None, j, :]
        denominators = np.sum(self.topic_prob, axis=0)
        denominators[denominators == 0] = 1
        self.topic_prob = self.topic_prob / denominators

        # M-step 
        # update P(z | d)
        for j in range(self.total_topics):
            self.document_topic_prob[:, j] = np.sum(np.multiply(self.term_doc_matrix, self.topic_prob[j]), axis=1)
        self.document_topic_prob /= np.sum(self.document_topic_prob, axis=1, keepdims=True) 
        
        # update P(w | z)
        for j in range(self.number_of_topics):
            self.topic_word_prob[j] = np.sum(np.multiply(self.term_doc_matrix, self.topic_prob[j]), axis=0)
            self.topic_word_prob[j] /= np.sum(self.topic_word_prob[j])
        # if there is a prior, then this part of the code will be executed.
        for j in range(self.number_of_topics, self.total_topics):
            self.topic_word_prob[j] = np.sum(np.multiply(self.term_doc_matrix, self.topic_prob[j]), axis=0) + self.mu * self.prior[j-self.number_of_topics]
            self.topic_word_prob[j] /= (np.sum(self.topic_word_prob[j]) + self.mu)
    
    def __calc_likelihood(self):
        """
        Calculate the log likelihood or the maximum a posteriori (MAP) estimation.
        """
        loglikelihood = np.sum(np.multiply(self.term_doc_matrix, np.log(np.matmul(self.document_topic_prob, self.topic_word_prob))))
        if self.prior is None:
            return loglikelihood
        expanded_prior = np.zeros([self.total_topics, self.vocabulary_size])
        expanded_prior[self.number_of_topics:] = self.prior
        logMAP = loglikelihood + self.mu * np.sum((np.multiply(expanded_prior, np.log(self.topic_word_prob))))
        return logMAP

    def __build_matrices(self):
        """
        Initialize the matrices with starting values, including normalization.
        If there is a prior, append the prior to the topic_word_prob matrix.
        """        
        # P(z | d, w)
        self.topic_prob = np.zeros([self.total_topics, self.number_of_documents, self.vocabulary_size], dtype=np.float)

        # P(z | d) 
        self.document_topic_prob = np.random.random(size = (self.number_of_documents, self.total_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        # P(w | z)
        self.topic_word_prob = np.random.random(size = (self.number_of_topics, self.vocabulary_size))
        if self.prior is not None:
            self.topic_word_prob.append(prior)
        self.topic_word_prob = normalize(self.topic_word_prob)

    def __run_algorithm(self):        
        # Run the EM algorithm and print the calculated likelihood or MAP
        for iteration in range(self.iterations):
            print("Iteration #" + str(iteration + 1) + "...")
            self.__EM_calc()
            print(self.__calc_likelihood())

    def initiate(self):
        """
        The first initial run of the PLSA program, build the document, vocabulary and term_doc matrix. 
        As the document, vocabulary, and the term_doc (word count) matrix are not changed by the prior, this method should only run one time. 
        """
        self.__build_document()
        self.__build_term_doc_matrix()
        self.__build_matrices()
        self.__run_algorithm()
            
    def calc_with_prior(self, prior):
        """
        After the initial run, the PLSA program will take in the prior and rerun the entire program with the prior added. 
        """
        self.prior = prior
        self.total_topics = self.number_of_topics + len(prior)
        self.__build_matrices()
        self.__run_algorithm()

def main():
    number_of_topics = 30
    max_iterations = 50
    mu = 30
    pl = plsa(number_of_topics, max_iterations, mu)    
    prior = None
    pl.initiate()

if __name__ == '__main__':
    main()