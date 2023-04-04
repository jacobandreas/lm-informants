import numpy as np
import matplotlib.pyplot as plt

def ngrams(seq, order):
    for i in range(len(seq) - order + 1):
        yield seq[i:i+order]

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def entropy(probs, length_norm=False):
    ent = -probs * np.log(probs) - (1-probs) * np.log(1-probs)
    if length_norm:
        return ent.sum()/len(ent)
    else:
        return ent.sum()


def plot_feature_probs(features, costs, last_costs, title=""):

    fig = plt.figure(figsize=(12, 5))

    if last_costs is None:
        colors = ["red" for _ in range(len(costs))]
    else:
        colors = ["blue" if c == lc else "red" for c, lc in zip(costs, last_costs)] 

    print("PLOTTING")
    print("costs: ", costs)
    print("last costs: ", last_costs)
    print("colors: ", colors)

    # Create the plot
    plt.clf()
    plt.scatter(features, costs, linestyle='None', marker='o', c=colors, s=3, alpha=0.5)
    plt.xlabel('Feature')
    plt.xticks(rotation=90)
    plt.ylim(-0.1, 1.1)
    plt.ylabel('Prob')
    plt.rc('xtick',labelsize=5)
    plt.title(title, fontsize=9.5)
    plt.tight_layout()
    plt.close()
    return fig
