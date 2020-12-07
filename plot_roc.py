
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from numpy import interp
from sklearn.metrics import roc_auc_score
import pickle
import os

fig = plt.figure(dpi=600)

for _f, l in [("test_eval_normal.pkl", "GPND"),
             ("test_eval.pkl", "GPND+J"),
             ("test_eval_pz_only.pkl", "GPND-parallel"),
             ("test_eval_perror_only.pkl", "GPND-perpendicular"),
              ]:

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(10):
        folder = "mnist_results_945/results_0_%d" % i

        with open(os.path.join(folder, _f), "rb") as f:
            y_scores, y_true = pickle.load(f)

        fpr[i], tpr[i], _ = roc_curve(y_true, y_scores)
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(10)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(10):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= 10

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr

    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    c = "macro"
    lw = 2
    plt.plot(fpr[c], tpr[c],
             lw=lw, label='%s (area = %0.3f)' % (l, roc_auc[c]))

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0, 0.4])
plt.ylim([0.6, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

fig.savefig("foo.pdf", bbox_inches='tight')
