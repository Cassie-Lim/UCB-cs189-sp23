import sys

if sys.version_info[0] < 3:
    raise Exception("Python 3 not detected.")
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, metrics


# partition dataset according to its type, return train set & val set
def make_dataset(data, label):
    data = data.reshape(len(data), -1)
    label = label.reshape(len(label), 1)
    dataset = np.concatenate([data, label], axis=-1)
    np.random.seed()
    np.random.shuffle(dataset)
    return dataset


if __name__ == "__main__":
    data_name = "spam"
    data = np.load(f"../data/{data_name}-data.npz")
    print("\n*********************************************************")
    print("\nloaded %s data!" % data_name)
    dataset = make_dataset(data['training_data'], data['training_labels'])
    # change k to implement k-fold
    k = 5
    l = int(len(dataset) / k)
    print("Training model for {}".format(data_name))

    train_acc_list = []
    val_acc_list = []
    train_num = 10000
    c_exp_list = range(-7, 0)
    # c_exp_list = range(-5, 5)
    # c_list = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    for c_pow in c_exp_list:
        c = pow(10, c_pow)
        print("Regularization param set to {} :".format(c))
        tmp_train_acc_list = []
        tmp_val_acc_list = []
        for i in range(k):
            print("\tRound {}".format(i+1))
            val_set = dataset[i*l: (i+1)*l, :]
            train_set = np.append(dataset[0:i * l, :], dataset[ (i+1)*l:-1, :], axis=0)
            train_x = train_set[:, :-1]
            train_y = train_set[:, -1]
            val_x = val_set[:, :-1]
            val_y = val_set[:, -1]

            model = svm.LinearSVC(dual=False, C=c)  # cancel dual to avoid 'ConvergenceWarning'
            model.fit(train_x[0:train_num, :], train_y[0:train_num])
            train_acc = metrics.accuracy_score(train_y, model.predict(train_x))
            val_acc = metrics.accuracy_score(val_y, model.predict(val_x))
            print("\t\tTrain acc: {}".format(train_acc))
            print("\t\tVal acc: {}".format(val_acc))
            tmp_train_acc_list.append(train_acc)
            tmp_val_acc_list.append(val_acc)
        avg_train_acc = np.average(tmp_train_acc_list)
        avg_val_acc = np.average(tmp_val_acc_list)
        train_acc_list.append(avg_train_acc)
        val_acc_list.append(avg_val_acc)
        print("\tAverage train acc: {}".format(avg_train_acc))
        print("\tAverage val acc: {}".format(avg_val_acc))
    plt.figure()
    plt.title('Train & val acc for {}'.format(data_name))
    plt.plot(c_exp_list, train_acc_list, 'bo', label='Training acc')
    plt.plot(c_exp_list, val_acc_list, 'b', label='Validation acc')
    plt.xlabel('Exp for c (base:10)')
    plt.ylabel('Acc')
    plt.legend()
    plt.savefig(data_name + '_' + str(k) + 'fold.png', bbox_inches='tight')
    plt.show()
