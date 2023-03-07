## Kaggle

### Overall accuarcy on Kaggle & Info

- Mnist: 0.98233
- Spam: 0.94666
- Cifar10: 0.597

> To speed up the training process, run on colab using gpu with thunderSVM. 

For spam test, use the vocabulary list from [Data2Science/vocab.txt at master · jetorz/Data2Science (github.com)](https://github.com/jetorz/Data2Science/blob/master/2020-04-26-ML-SVM/vocab.txt), and append new features by running `featurize.py`

> Note: It is said we can achieve better performance by deleting some features that are less predictive. Try it out! :)

For cifar test,  I use hog features, LBP features and color features. Feel free to add more helpful stuffs!

