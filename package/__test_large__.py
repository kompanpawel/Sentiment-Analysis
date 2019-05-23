import json

import pandas as pd
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import tokenizer_from_json
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt

loaded_CNN_model = load_model("CNN_best_weights_3.03-0.8318.hdf5")
with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)


def checktweet(tweet):
    sequence = tokenizer.texts_to_sequences([tweet])
    x_test_seq = pad_sequences(sequence, maxlen=45)
    yhat_cnn = loaded_CNN_model.predict(x_test_seq)
    return yhat_cnn


def calculate_accuracy(value):
    if value < 0.5:
        return (0.5 - value) * 200
    else:
        return (value - 0.5) * 200


df = pd.read_csv("../testingData/testdata.manual.2009.06.14.csv", header=None, usecols=[0, 5],
                 names=['sentiment', 'text'], encoding="ISO-8859-1")

df = df.drop(df[df.sentiment == 2].index, axis=0)
df = df.reset_index(drop=True)
df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})
correct = 0
percent_accuracy = 0
for i in range(0, df.shape[0]):
    check = checktweet(df['text'][i])
    if check < 0.5:
        pos_neg = 0
    else:
        pos_neg = 1
    accuracy = calculate_accuracy(check)
    percent_accuracy += accuracy
    if pos_neg == df['sentiment'][i]:
        correct += 1

accuracy = correct / df.shape[0]
percent = percent_accuracy / df.shape[0]
print(accuracy, percent)

y_test = df['sentiment']
sequences = tokenizer.texts_to_sequences(df['text'])
test_seq = pad_sequences(sequences, maxlen=45)
y_cnn = loaded_CNN_model.predict(test_seq)
fpr_cnn, tpr_cnn, treshold = roc_curve(y_test, y_cnn)
roc_auc_nn = auc(fpr_cnn, tpr_cnn)
plt.figure(figsize=(8, 7))
plt.plot(fpr_cnn, tpr_cnn, label='CNN (area=%0.3f)' % roc_auc_nn, linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('Receiver  ', fontsize=18)
plt.legend(loc="lower right")
plt.show()
