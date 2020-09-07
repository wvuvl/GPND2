from utils import cifar_reader
from utils.download import download
import random
import pickle

download(directory="CIFAR-10", url="https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz", extract_targz=True)

items_train = cifar_reader.Reader('CIFAR-10/cifar-10-batches-bin', train=True, test=True).items

folds = 5

#Split mnist into 5 folds:
class_bins = {}

random.shuffle(items_train)

for x in items_train:
    if x[0] not in class_bins:
        class_bins[x[0]] = []
    class_bins[x[0]].append(x)

cifar_folds = [[] for _ in range(folds)]

for _class, data in class_bins.items():
    count = len(data)
    print("Class %d count: %d" % (_class, count))

    count_per_fold = count // folds

    for i in range(folds):
        cifar_folds[i] += data[i * count_per_fold: (i + 1) * count_per_fold]


print("Folds sizes:")
for i in range(len(cifar_folds)):
    print(len(cifar_folds[i]))

    output = open('data_fold_%d.pkl' % i, 'wb')
    pickle.dump(cifar_folds[i], output)
    output.close()

