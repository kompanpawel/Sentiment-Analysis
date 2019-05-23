import json

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import tokenizer_from_json


def calculateAccuracy(value):
    if value < 0.5:
        return (0.5 - value) * 200
    else:
        return (value - 0.5) * 200


loaded_CNN_model = load_model("CNN_best_weights_3.03-0.8318.hdf5")
# tokenizer = Tokenizer(num_words=100000)

tweet = ['This movie is not good']

# tokenizer.fit_on_texts(x)
# tokenizer_json = tokenizer.to_json()
# with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
#     f.write(json.dumps(tokenizer_json, ensure_ascii=False))

with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

sequences_test = tokenizer.texts_to_sequences(tweet)
x_test_seq = pad_sequences(sequences_test, maxlen=45)
yhat_cnn = loaded_CNN_model.predict(x_test_seq)
result = []
i = 0
for predict in yhat_cnn:
    if predict < 0.5:
        result.append("\"%s\" - negative, accuracy = %d%%" % (tweet[i], calculateAccuracy(predict)))
    else:
        result.append("\"%s\" - positive, accuracy = %d%%" % (tweet[i], calculateAccuracy(predict)))
    i += 1

for i in result:
    print(i)

