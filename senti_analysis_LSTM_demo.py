import numpy as np
import gensim
import time
import nltk
import tensorflow as tf
from tensorflow.contrib import slim
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize


label_map = ['negative', 'neutral', 'positive']

weight_file = './weight-e-35_256_2000_64_1_535-0.660.ckpt'

# input_sentence = input('type the sentence\t>>')
input_sentence = 'We avoided delisting of the \'Samba\' list, but...\"It\'s not a fraudulent accounting pardon.\"'

input_dim = 300
num_unit = 64
maxSeqLength = 25
num_class = 3
drop_out = 0.5

t = time.time()
print('load pre-trained word2vec dict...', end=' ')
word2vec = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
print('End {:.2f}sec'.format(time.time() - t))
stemmer = SnowballStemmer(language='english')
stopword = nltk.corpus.stopwords.words('english')


# Functions
def rempct(s):
    punct_numb = '\"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~€$' + '0123456789' + "\'"
    swopct = ""
    for letter in s:
        if letter not in punct_numb:
            swopct += letter.lower()
        else:
            swopct += ' '
    return swopct


def cleaner(txt):
    txt = str(txt.encode('ascii', 'ignore'))
    txt = txt.replace("\\n"," ")
    txt = txt.replace("\n"," ")
    txt = txt.replace('b\'',"")
    txt = txt.replace('b\"',"")
    return txt


def stemming(sent):
    stemmed = []
    for i, token in enumerate(sent):
        stem = stemmer.stem(token)
        if stem not in stopword:
            stemmed.append(stem)
    return stemmed


def fc(x, num_out, activation):
    fc_layer = slim.fully_connected(inputs=x, num_outputs=num_out, activation_fn=activation)
    return fc_layer


def zero_padding(sentences):
    batch = np.shape(sentences)[0]
    input_mat = np.zeros([batch, maxSeqLength, input_dim], dtype=np.float32)
    for i, sentence in enumerate(sentences):
        for j, vec in enumerate(sentence):
            if j == maxSeqLength - 1:
                break
            input_mat[i][j] = vec
    return input_mat


x = tf.placeholder(dtype=tf.float32, shape=[None, maxSeqLength, input_dim]) # [batch, 25, 300]
y = tf.placeholder(dtype=tf.int64, shape=[None])    # [batch]
y_onehot = tf.one_hot(indices=y, depth=num_class)   # [batch , 3]
onehot_shape = tf.shape(y_onehot)

lstmCell = [tf.nn.rnn_cell.BasicLSTMCell(num_units=num_unit) for _ in range(3)]
lstmCell = tf.nn.rnn_cell.MultiRNNCell(lstmCell)
lstmCell = tf.nn.rnn_cell.DropoutWrapper(cell=lstmCell, output_keep_prob=drop_out)
val, _ = tf.nn.dynamic_rnn(cell=lstmCell, inputs=x, dtype=tf.float32)   # val: [batch, num_words, num_unit]
val_trans = tf.transpose(val, [1, 0, 2])
output = tf.gather(val_trans, int(val_trans.get_shape()[0]) -1)

logit = fc(output, num_out=num_class, activation=None)
y_pred = tf.nn.softmax(logit)
y_pred_cls = tf.argmax(y_pred, dimension=1)


var = [var for var in tf.all_variables()]
saver = tf.train.Saver(var_list=var)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    saver.restore(sess=sess, save_path=weight_file)

    txt = cleaner(input_sentence)
    sent = txt
    # print('cleaner: ', txt)
    txt = rempct(txt)
    # print('rempct: ', txt)

    # Tokenize -> Words
    txt = word_tokenize(txt)
    # print('tokenizing: ', txt)
    txt = stemming(txt)
    # print('stemming: ', txt)

    # Number of words in article
    nwords = len(txt)
    words = []
    vectors = []
    # ------------------ End Pre-processing ------------------
    # ------------------ Applying Word2Vec -------------------
    for word in txt:
        try:
            vector = np.asarray(word2vec[word], dtype=np.float32)
            words.append(word)
            vectors.append(vector)
            # print('{}\t{}'.format(word, np.shape(vector)))
        except KeyError:
            print('Word {} is not in vocabulary'.format(word))
    vectors = np.expand_dims(vectors, axis=0)

    X = zero_padding(vectors)
    pred = sess.run(y_pred_cls, feed_dict={x: X})

    print('Input sentence: {}\nPrediction: {}'.format(input_sentence, label_map[pred[0]]))
