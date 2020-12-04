import numpy as np
import math


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
        self.vocabulary = []
        self.likelihoods = []
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
        f = open(self.documents_path, "r")
        for line in f:
            words_in_line = []
            words_in_line = line.split()
            try: 
                words_in_line.remove('0')
            except ValueError:
                pass #
            try: 
                words_in_line.remove('1')
            except ValueError:
                pass #

            self.documents.append(words_in_line)
            self.number_of_documents += 1
        f.close()
        

    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus. Put it in self.vocabulary
        for example: ["rain", "the", ...]

        Update self.vocabulary_size
        """      
        self.words_in_doc = {}  
        for document in self.documents:
            for word in document:
                if word in self.words_in_doc:
                    self.words_in_doc[word] += 1
                else:
                    self.words_in_doc[word] = 1
                    self.vocabulary.append(word)

        self.vocabulary_size = len(self.words_in_doc)


    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document, 
        and each column represents a vocabulary term.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """
        self.term_doc_matrix = np.zeros([self.number_of_documents, self.vocabulary_size], np.int8)
        for vocab in range(self.vocabulary_size):
            for line in range(self.number_of_documents):
                for word in self.documents[line]:
                    if word == self.vocabulary[vocab]:
                        self.term_doc_matrix[line][vocab] += 1
        

    def initialize_randomly(self, number_of_topics):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob

        Don't forget to normalize! 
        HINT: you will find numpy's random matrix useful [https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.random.html]
        """
        self.document_topic_prob = np.random.random(size = (self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.random.random(size = (number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = normalize(self.topic_word_prob) 
        

    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform 
        probability distribution. This is used for testing purposes.

        DO NOT CHANGE THIS FUNCTION
        """
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.ones((number_of_topics, len(self.vocabulary)))
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
        """ print("E step:")
        for i in range(self.number_of_documents):
            for j in range(self.vocabulary_size):
                denominator = 0;
                for k in range(number_of_topics):
                    self.topic_prob[i, j, k] = self.topic_word_prob[k, j] * self.document_topic_prob[i, k];
                    denominator += self.topic_prob[i, j, k];
                if denominator == 0:
                    for k in range(number_of_topics):
                        self.topic_prob[i, j, k] = 0;
                else:
                    for k in range(number_of_topics):
                        self.topic_prob[i, j, k] /= denominator; """
        for j in range(number_of_topics): 
            self.topic_prob[j] = np.expand_dims(self.document_topic_prob[:, j], 1) * np.expand_dims(self.topic_word_prob[j, :], 0)
        self.topic_prob = self.topic_prob / np.sum(self.topic_prob, axis=0)

    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z)
        """
        """ print("M step:")        
        # update P(w | z)
        for k in range(number_of_topics):
            denominator = 0
            for j in range(self.vocabulary_size):
                self.topic_word_prob[k, j] = 0
                for i in range(self.number_of_documents):
                    self.topic_word_prob[k, j] += self.term_doc_matrix[i, j] * self.topic_prob[i, j, k]
                denominator += self.topic_word_prob[k, j]
            if denominator == 0:
                for j in range(self.vocabulary_size):
                    self.topic_word_prob[k, j] = 1.0 / M
            else:
                for j in range(self.vocabulary_size):
                    self.topic_word_prob[k, j] /= denominator """
        for j in range(self.number_of_topics):
            self.topic_word_prob[j] = np.sum(np.multiply(self.term_doc_matrix, self.topic_prob[j]), axis=0)
            self.topic_word_prob[j] /= np.sum(self.topic_word_prob[j])
        
        """ # update P(z | d)
        for i in range(self.number_of_documents):
            for k in range(number_of_topics):
                self.document_topic_prob[i, k] = 0
                denominator = 0
                for j in range(self.vocabulary_size):
                    self.document_topic_prob[i, k] += self.term_doc_matrix[i, j] * self.topic_prob[i, j, k]
                    denominator += self.term_doc_matrix[i, j];
                if denominator == 0:
                    self.document_topic_prob[i, k] = 1.0 / K
                else:
                    self.document_topic_prob[i, k] /= denominator """
        for j in range(self.number_of_topics):
            self.document_topic_prob[:, j] = np.sum(np.multiply(self.term_doc_matrix, self.topic_prob[j]), axis=1)
            self.document_topic_prob[:, j] /= np.sum(self.document_topic_prob[:, j]) 

    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        
        Append the calculated log-likelihood to self.likelihoods

        """
        """ loglikelihood = 0
        for i in range(self.number_of_documents):
            for j in range(self.vocabulary_size):
                tmp = 0
                for k in range(number_of_topics):
                    tmp += self.topic_word_prob[k, j] * self.document_topic_prob[i, k]
                if tmp > 0:
                    loglikelihood += term_doc_matrxi[i, j] * math.log(tmp)        
        self.likelihoods.append(loglikelihood) """
                

    def plsa(self, number_of_topics, max_iter, epsilon):

        """
        Model topics.
        """
        print ("EM iteration begins...")
        
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


def main():
    documents_path = 'data/test.txt'
    corpus = Corpus(documents_path)  # instantiate corpus
    corpus.build_corpus()
    corpus.build_vocabulary()
    print(corpus.vocabulary)
    print("Vocabulary size:" + str(len(corpus.vocabulary)))
    print("Number of documents:" + str(len(corpus.documents)))
    number_of_topics = 2
    max_iterations = 50
    epsilon = 0.001
    mu = 30
    prior = None
    corpus.plsa(number_of_topics, max_iterations, epsilon, mu, prior)


if __name__ == '__main__':
    main()
