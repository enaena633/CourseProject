""" Coded by: Jack Ma, Ena Yu, and Matt McCarty, Team JEM, University of Illinois at Urbana Champaign
CS 410 Text Information System Fall 2020 Final Project
"""
import numpy as np

def calc_prior(significance):
    """ 
    Input an array of impact and significance. Positive significance means positive impact and vice versa. 
    """
    prior = []

    # separate the significance according to their impact
    cutoff = .5
    positive_sigs = significance - cutoff
    negative_sigs = -significance - cutoff
    positive_sigs[positive_sigs<0] = 0
    negative_sigs[negative_sigs<0] = 0

    # if one orientation is very weak (<10%) then ignore that group
    # otherwise append that group into the return
    for i in range(len(positive_sigs)):
        pos_orientation = np.count_nonzero(positive_sigs[i])
        neg_orientation = np.count_nonzero(negative_sigs[i])
        tot = pos_orientation + neg_orientation
        if tot != 0:
            pos_percent = pos_orientation / tot
            if (pos_percent < 0.1):
                prior.append(negative_sigs[i]/np.sum(negative_sigs[i]))
            elif (pos_percent > 0.9):
                prior.append( positive_sigs[i]/np.sum(positive_sigs[i]))
            else:
                prior.append(positive_sigs[i]/np.sum(positive_sigs[i]))
                prior.append(negative_sigs[i]/np.sum(negative_sigs[i]))                
    
    if not prior:
        return None


    return np.asarray(prior)
