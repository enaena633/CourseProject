""" 
Coded by: Jack Ma, Ena Yu, and Matt McCarty, Team JEM, University of Illinois at Urbana Champaign
CS 410 Text Information System Fall 2020 Final Project
"""
import numpy as np
import pandas as pd
import plsawithprior as plsa
from Granger_Casuality_Test import granger_test as granger
import calc_prior as cp

def main():
    # Define the parameters of the experiment. Defaults to Tn = 30, Mu = 30, and 50 PLSA iterations to converge the result.
    number_of_topics = 20
    max_iterations = 10
    mu = 30
    pl = plsa.plsa(number_of_topics, max_iterations, mu)    
    prior = None
    pl.initiate()

    # Run the granger test and feed back the prior to PLSA 5 times. 
    time_series_data = pd.read_csv("time_series.csv",sep=',')
    for i in range(5):
        topics = []
        #first pick significant topics
        for j in range(pl.total_topics):
            topic_significance = np.absolute(granger(time_series_data, pl.document_topic_prob[:, j]))
            topics.append(topic_significance)

        top_topics = np.argpartition(topics, -5)[-5:]


        #get word significance from the top ten topics
        sig_array = []
        for top_topic in top_topics:
            sig_array_per_topic = np.zeros(pl.vocabulary_size)
            topic_word_prob_dist = pl.topic_word_prob[top_topic,:]
            for m in np.argpartition(topic_word_prob_dist, -100)[-100:]:
                word_stream = pl.term_doc_matrix[:, m]
                word_significance = granger(time_series_data, word_stream)
                sig_array_per_topic[m] = word_significance
            sig_array.append(sig_array_per_topic)            

        prior = cp.calc_prior(np.asarray(sig_array))
        print('With prior number #' + str(i+1))

        pl.calc_with_prior(prior)
    
    # Run the granger one final time to find the ten top significant topics ater 5 iterations.
    topics = []
    #first pick significant topics
    for j in range(pl.total_topics):
        topic_significance = np.absolute(granger(time_series_data, pl.document_topic_prob[:, j]))
        topics.append(topic_significance)
        top_topics = np.argpartition(topics, -number_of_top_topics)[-number_of_top_topics:]

    #get word significance from the top ten topics
    sig_array = []
    for top_topic in top_topics:
        sig_array_per_topic = []
        for m in range(pl.vocabulary_size):
            word_stream = np.multiply(pl.term_doc_matrix[:, m], pl.topic_word_prob[top_topic, m])
            word_significance = granger(time_series_data, word_stream)
            sig_array_per_topic.append(word_significance)
        sig_array.append(sig_array_per_topic)            
    prior = cp.calc_prior(np.asarray(sig_array))
    
    for topic in prior:
        top_words = np.argpartition(topic, -3)[-3:]
        for word in top_words:
            print(pl.vocabulary.get(word), end = ' ')
        print('')

if __name__ == '__main__':
    main()
