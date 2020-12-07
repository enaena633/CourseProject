import numpy as np

""" input two array of x by y where x is the number of topics and y is the words in the vocabulary
 """
def calc_prior(significance):
    prior = []
    # separate the significance according to their impact
    positive_sigs = significance- 95
    negative_sigs = -significance - 95
    positive_sigs[positive_sigs<0] = 0
    negative_sigs[negative_sigs<0] = 0

    # if one orientation is very weak (<10%) then ignore that group
    # otherwise append that group into the return
    for i in range(len(positive_sigs)):
            pos_orientation = np.count_nonzero(positive_sigs[i])
            neg_orientation = np.count_nonzero(negative_sigs[i])
            tot = pos_orientation + neg_orientation
            pos_percent = pos_orientation / tot
            if (pos_percent < 0.1):
               prior.append(negative_sigs[i]/np.sum(negative_sigs[i]))
            elif (pos_percent > 0.9):
                prior.append(positive_sigs[i]/np.sum(positive_sigs[i]))
            else:
                prior.append(positive_sigs[i]/np.sum(positive_sigs[i]))
                prior.append(negative_sigs[i]/np.sum(negative_sigs[i]))
                
    return prior


def main():
    significance = np.array([[99.0, 96.0, 2.0, -99.0, -97.0], [99.0, 97.0, -1.0, -96.0, -96.0]])
    prior = calc_prior( significance)
    print(prior)

if __name__ == '__main__':
    main()