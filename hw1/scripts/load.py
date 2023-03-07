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
    for data_name in ["mnist", "spam", "cifar10"]:
        data = np.load(f"../data/{data_name}-data.npz")
        print("\n*********************************************************")
        print("\nloaded %s data!" % data_name)
        fields = "test_data", "training_data", "training_labels"
        for field in fields:
            print(field, data[field].shape)
        train_set, val_set = do_partition(data_name, data['training_data'], data['training_labels'])
        train_x = train_set[:, :-1]
        train_y = train_set[:, -1]
        val_x = val_set[:, :-1]
        val_y = val_set[:, -1]
        if data_name == "mnist":
            train_num_list = [100, 200, 500, 1000, 2000, 5000, 10000]
        elif data_name == "spam":
            train_num_list = [100, 200, 500, 1000, 2000, len(train_y)]
        else:
            train_num_list = [100, 200, 500, 1000, 2000, 5000]
        model = svm.LinearSVC(dual=False)  # cancel dual to avoid 'ConvergenceWarning'
        print("Training model for {}".format(data_name))
        train_acc_list = []
        val_acc_list = []
        for train_num in train_num_list:
            print("Using {} training examples:".format(train_num))
            model.fit(train_x[0:train_num, :], train_y[0:train_num])
            train_acc = metrics.accuracy_score(train_y, model.predict(train_x))
            val_acc = metrics.accuracy_score(val_y, model.predict(val_x))
            print("\tTrain acc: {}".format(train_acc))
            print("\tVal acc: {}".format(val_acc))
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
        plt.figure()
        plt.title('Train & val acc for {}'.format(data_name))
        plt.plot(train_num_list, train_acc_list, 'bo', label='Training acc')  # 'bo'为画蓝色圆点，不连线
        plt.plot(train_num_list, val_acc_list, 'b', label='Validation acc')
        plt.legend()
        plt.savefig(data_name+'.png', bbox_inches='tight')
        plt.show()
