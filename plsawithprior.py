import numpy as np
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

    def __init__(self, vocabulary, number_of_topics, max_iterations, mu):
        """
        Initialize empty document list.
        """
        self.documents = []
        self.vocabulary = vocabulary
        self.term_doc_matrix = None 
        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z)
        self.topic_prob = None  # P(z | d, w)
        self.number_of_documents = 0
        self.vocabulary_size = 0
        self.number_of_topics = number_of_topics
        self.total_topics = 0
        self.mu = mu
        self.prior = None
        self.iterations = max_iterations

    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document, 
        and each column represents a vocabulary term.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """
        self.term_doc_matrix = np.zeros([self.number_of_documents, self.vocabulary_size], np.int8)
        for doc in range(self.number_of_documents):
            for word in self.vocabulary.doc2bow(self.documents[doc]):
                self.term_doc_matrix[doc, word[0]] = word[1]
            
    def EM_calc(self):
        # E-step updates P(z | w, d)        
        for j in range(self.total_topics): 
            self.topic_prob[j] = np.expand_dims(self.document_topic_prob[:, j], 1) * np.expand_dims(self.topic_word_prob[j, :], 0)
        denominators = np.sum(self.topic_prob, axis=0)
        denominators[denominators == 0] = 1
        self.topic_prob = self.topic_prob / denominators

        # M-step update P(z | d)
        for j in range(self.total_topics):
            self.document_topic_prob[:, j] = np.sum(np.multiply(self.term_doc_matrix, self.topic_prob[j]), axis=1)
            self.document_topic_prob[:, j] /= np.sum(self.document_topic_prob[:, j]) 
        
        # update P(w | z)
        for j in range(self.number_of_topics):
            self.topic_word_prob[j] = np.sum(np.multiply(self.term_doc_matrix, self.topic_prob[j]), axis=0)
            self.topic_word_prob[j] /= np.sum(self.topic_word_prob[j])
        for j in range(self.number_of_topics, self.total_topics):
            self.topic_word_prob[j] = np.sum(np.multiply(self.term_doc_matrix, self.topic_prob[j]), axis=0) + self.mu * prior[j-self.number_of_topics]
            self.topic_word_prob[j] /= (np.sum(self.topic_word_prob[j]) + self.mu)
    

    def calc_likelihood(self):
        loglikelihood = np.sum(np.multiply(self.term_doc_matrix, np.log(np.matmul(self.document_topic_prob, self.topic_word_prob) + 1.0)))
        if self.mu == 0:
            return loglikelihood
        expanded_prior = np.zeros([self.total_topics, self.vocabulary_size])
        expanded_prior[self.number_of_topics:] = self.prior
        logMAP = loglikelihood + self.mu * np.sum((np.multiply(expanded_prior, np.log(self.topic_word_prob + 1.0) )))
        return logMAP

    def initiate(self, documents, prior):

        """
        Model topics.
        """
        print (datetime.now())

        self.total_topics = self.number_of_topics

        # determine if there is prior
        if prior is None:
            self.mu = 0

        else:
            self.mu = mu
            self.prior = prior
            self.total_topics += len(prior)

        self.documents = documents
        self.number_of_documents = len(self.documents)
        self.vocabulary_size = len(self.vocabulary)
        
        # build term-doc matrix
        self.build_term_doc_matrix()
        
        # P(z | d, w)
        self.topic_prob = np.zeros([self.total_topics, self.number_of_documents, self.vocabulary_size], dtype=np.float)
        print(self.topic_prob)

        # P(z | d) 
        self.document_topic_prob = np.random.random(size = (self.number_of_documents, self.total_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)
        print(self.document_topic_prob)

        # P(w | z)
        self.topic_word_prob = np.random.random(size = (self.total_topics, self.vocabulary_size))
        if prior is not None:
            self.topic_word_prob.append(prior)
        self.topic_word_prob = normalize(self.topic_word_prob)
        print(self.topic_word_prob)

        # Run the EM algorithm
        current_likelihood = 0.0

        for iteration in range(self.iterations):
            print("Iteration #" + str(iteration + 1) + "...")
            self.EM_calc()
            print(self.calc_likelihood())
            
        print(datetime.now())

def build_vocabulary():
    documents = []
    with open('sanitized_nyt.tsv') as f:
        documents = f.read().splitlines()
    documents = preprocess_documents(documents)
    vocabulary = corpora.Dictionary(documents)
    vocabulary_size = len(vocabulary)
    print("Vocabulary size:" + str(vocabulary_size))
    return vocabulary

def build_document():
    documents = []
    with open('sanitized_nyt2.tsv') as f:
        documents = f.read().splitlines()
    documents = preprocess_documents(documents)

    print("Number of documents:" + str(len(documents)))
    return documents

def main():
    vocabulary = build_vocabulary()
    number_of_topics = 30
    max_iterations = 50
    mu = 30
    pl = plsa(vocabulary, number_of_topics, max_iterations, mu)
    documents = build_document()
    prior = None
    pl.initiate(documents, prior)

if __name__ == '__main__':
    main()