import sys

if sys.version_info[0] < 3:
    raise Exception("Python 3 not detected.")
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from scipy import io


# partition dataset according to its type, return train set & val set
def do_partition(data_name, data, label):
    data = data.reshape(len(data), -1)
    label = label.reshape(len(label), 1)
    dataset = np.concatenate([data, label], axis=-1)
    np.random.seed()
    np.random.shuffle(dataset)
    if data_name == 'mnist':
        index = 10000
    elif data_name == 'spam':
        index = int(len(dataset) * 0.2)
    else:
        index = 5000
    val_set = dataset[0:index]
    train_set = dataset[index:-1]
    return train_set, val_set


if __name__ == "__main__":
    data_name = "mnist"
    data = np.load(f"../data/mnist-data.npz")
    print("\n*********************************************************")
    print("\nloaded %s data!" % data_name)
    train_set, val_set = do_partition(data_name, data['training_data'], data['training_labels'])
    train_x = train_set[:, :-1]
    train_y = train_set[:, -1]
    val_x = val_set[:, :-1]
    val_y = val_set[:, -1]

    print("Training model for {}".format(data_name))
    train_acc_list = []
    val_acc_list = []
    train_num = 10000
    c_exp_list = range(-5, 5)
    # c_list = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    for c_pow in c_exp_list:
        c = pow(10, c_pow)
        model = svm.LinearSVC(dual=False, C=c)  # cancel dual to avoid 'ConvergenceWarning'
        print("Regularization param set to {} :".format(c))
        model.fit(train_x[0:train_num, :], train_y[0:train_num])
        train_acc = metrics.accuracy_score(train_y, model.predict(train_x))
        val_acc = metrics.accuracy_score(val_y, model.predict(val_x))
        print("\tTrain acc: {}".format(train_acc))
        print("\tVal acc: {}".format(val_acc))
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
    plt.figure()
    plt.title('Train & val acc for {}'.format(data_name))
    plt.plot(c_exp_list, train_acc_list, 'bo', label='Training acc')
    plt.plot(c_exp_list, val_acc_list, 'b', label='Validation acc')
    plt.xlabel('Exp for c (base:10)')
    plt.ylabel('Acc')
    plt.legend()
    plt.savefig(data_name+'_param_tune.png', bbox_inches='tight')
    plt.show()
