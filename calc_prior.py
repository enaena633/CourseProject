import numpy as np

""" input two array of x by y where x is the number of topics and y is the words in the vocabulary
 """
def calc_prior(impact, significance):
    prior = []
    impacts = impact
    impacts[impacts==0] = -1
    sigs = np.multiply(significance, impacts)
    positive_sigs = sigs - 95
    negative_sigs = (sigs + 95) * -1
    all_sigs = np.concatenate((positive_sigs, negative_sigs))
    all_sigs[all_sigs<0] = 0
    for topic in all_sigs:
        sum = np.sum(topic)
        if sum > 0:
            prior.append(topic/sum)    
    return prior

def main():
    impact = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 0, 1]])
    significance = np.array([[99.0, 96.0, 2.0, 99.0, 97.0], [99.0, 97.0, 1.0, 96.0, 96.0]])
    prior = calc_prior(impact, significance)
    print(prior)

if __name__ == '__main__':
    main()