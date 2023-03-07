import sys

if sys.version_info[0] < 3:
    raise Exception("Python 3 not detected.")
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from scipy import io
from save_csv import results_to_csv

# partition dataset according to its type, return train set & val set
def do_partition(data_name, data, label):
    # data = data.reshape(len(data), -1)
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


def do_standardize(test_data, train_data):
    test_data = test_data.reshape(len(test_data), -1)
    train_data = train_data.reshape(len(train_data), -1)
    index = len(test_data)
    print(index)
    data = np.append(test_data, train_data, axis=0)
    data = (data - np.min(data))/(np.max(data) - np.min(data))
    return data[0: index, :], data[index::, :]


if __name__ == "__main__":
    data_name = "mnist"
    data = np.load(f"../data/mnist-data.npz")
    print("\n*********************************************************")
    print("\nloaded %s data!" % data_name)
    test_data = data["test_data"]
    train_data = data['training_data']
    test_data, train_data = do_standardize(test_data, train_data)
    print(len(test_data))
    train_set, val_set = do_partition(data_name, train_data, data['training_labels'])
    train_x = train_set[:, :-1]
    train_y = train_set[:, -1]
    val_x = val_set[:, :-1]
    val_y = val_set[:, -1]

    print("Training model for {}".format(data_name))
    train_acc_list = []
    val_acc_list = []
    train_num = len(train_data)
    # c_exp_list = range(-5, 5)
    # c=0.02, best performance for linear
    c_list = [1.0, 10, 100]
    gamma_list = [1, 0.1, 0.01, 0.001]
    # c_list = [0.01, 0.015, 0.02]
    for c in c_list:
        for gamma in gamma_list:
            # model = svm.LinearSVC(dual=False, C=c)  # cancel dual to avoid 'ConvergenceWarning'
            model = svm.SVC(C=c, gamma=gamma)
            print("Regularization param {}, gamma {} :".format(c, gamma))
            model.fit(train_x[0:train_num, :], train_y[0:train_num])
            train_acc = metrics.accuracy_score(train_y, model.predict(train_x))
            val_acc = metrics.accuracy_score(val_y, model.predict(val_x))
            print("\tTrain acc: {}".format(train_acc))
            print("\tVal acc: {}".format(val_acc))
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
            results_to_csv(model.predict(test_data), data_name + "_" + str(c) + "_" + str(gamma))
    plt.figure()
    plt.title('Train & val acc for {}'.format(data_name))
    plt.plot(c_list, train_acc_list, 'bo', label='Training acc')
    plt.plot(c_list, val_acc_list, 'b', label='Validation acc')
    plt.xlabel('Exp for c (base:10)')
    plt.ylabel('Acc')
    plt.legend()
    # plt.savefig(data_name+'_param_tune.png', bbox_inches='tight')
    plt.show()
