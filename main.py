""" 
Coded by: Jack Ma, Ena Yu, and Matt McCarty, Team JEM, University of Illinois at Urbana Champaign
CS 410 Text Information System Fall 2020 Final Project
"""
import numpy as np
import plsawithprior as plsa
import Granger_Casuality_Test as granger
import calc_prior as cp

def main():
    # Define the parameters of the experiment. Defualts to Tn = 30, Mu = 30, and 50 PLSA iterations to converge the result.
    number_of_topics = 30
    max_iterations = 50
    mu = 30
    pl = plsa.plsa(number_of_topics, max_iterations, mu)    
    prior = None
    pl.initiate()

    # Run the granger test and feedback back the prior to PLSA 5 times. 
    for i in range(5):
        granger_result = granger(pl.document_topic_prob, pl.topic_word_prob, pl.term_doc_matrix)
        prior = cp.calc_prior()
        pl.calc_with_prior(prior)
    
    # Run the granger one final time to find the ten top significant topics ater 5 iterations.
    granger_result = granger(pl.document_topic_prob, pl.topic_word_prob, pl.term_doc_matrix)
    prior = cp.calc_prior()
    for topic in prior:
        top_words = np.argpartition(topic, -3)[-3:]
        for word in top_words:
            print(pl.vocabulary.get(word), end = ' ')
        print('')

if __name__ == '__main__':
    main()