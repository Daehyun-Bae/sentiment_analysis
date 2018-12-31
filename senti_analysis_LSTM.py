import numpy as np
import os
import pickle
import argparse
import tensorflow as tf
from tensorflow.contrib import slim

"""
2018.12.20
Sentiment analysis using LSTM
"""
parse = argparse.ArgumentParser()
parse.add_argument('--lr', type=float, default=0.005, help='the learning rate')
parse.add_argument('--epoch', type=int, default=2000, help='the number of epochs')
parse.add_argument('--batch', type=int, default=256, help='the batch size')
parse.add_argument('--gpu', type=int, default=0, help='Select gpu number. Default is 0')
parse.add_argument('--testnum', type=str, default='e-35_256_2000_1', help='Prefix of test name')
parse.add_argument('--h_unit', type=int, default=64, help='the number of hidden unit')

label_map = ['negative', 'neutral', 'positive']
args = parse.parse_args()
data_dic = ''
data_file = ''      # define path to pre-embedded dataset
data_path = os.path.join(data_dic, data_file)
test_num = args.testnum
gpu = args.gpu
batch_size = args.batch
epoch = args.epoch
num_unit = args.h_unit
lr = args.lr

input_dim = 300
maxSeqLength = 25
num_class = 3
drop_out = 0.5


def load_data(data_path):
    # Load data...
    # data structure
    # data(dict):
    # ['words']: (list) list of words per sentence
    # ['vectors']: (list) list of vectors per sentence (300 dim)
    # ['label']: (list) list of labels per sentence (0: neg 1: nue 2: pos)
    # ['sentences']: (list) list of raw data.

    with open(data_path, 'rb') as pkl:
        load_dict = pickle.load(pkl)

    words = np.asarray(load_dict['words'])
    vectors = np.asarray(load_dict['vectors'])
    labels = np.asarray(load_dict['label'], dtype=np.int64)
    sentences = np.asarray(load_dict['sentences'])
    return [words, vectors, labels, sentences]


WordList, VectorList, LabelList, SentList = load_data(data_path)
total_data = len(LabelList)
print('Total data: ', total_data)

label_tr = []
word_tr = []
vec_tr = []

label_ts = []
word_ts = []
vec_ts = []
sent_ts = []
sampling = np.random.random_sample(total_data)

# Split dataset into training and test
for i, prob in enumerate(sampling):
    if prob <= .8:
        label_tr.append(LabelList[i])
        word_tr.append(WordList[i])
        vec_tr.append(VectorList[i])
    else:
        label_ts.append(LabelList[i])
        word_ts.append(WordList[i])
        vec_ts.append(VectorList[i])
        sent_ts.append(SentList[i])
print('Tr / Ts: {} / {}'.format(len(label_tr), len(label_ts)))


# Remove the biased class 'neutral'
sampling = np.random.random_sample(len(label_tr))
label_tmp = []
word_tmp = []
vec_tmp = []

for i, prob in enumerate(sampling):
    if prob <= .5 and label_tr[i] == 1:
        continue
    label_tmp.append(label_tr[i])
    word_tmp.append(word_tr[i])
    vec_tmp.append(vec_tr[i])
label_tr = label_tmp
word_tr = word_tmp
vec_tr = vec_tmp

neg = neu = pos = 0
for label in label_tr:
    if label == 0:
        neg += 1
    elif label == 1:
        neu += 1
    else:
        pos += 1
print('TRAIN\t\tNeg {}\tNeu {}\tPos {}'.format(neg, neu, pos))

neg = neu = pos = 0
for label in label_ts:
    if label == 0:
        neg += 1
    elif label == 1:
        neu += 1
    else:
        pos += 1
print('TEST\t\tNeg {}\tNeu {}\tPos {}'.format(neg, neu, pos))


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


x = tf.placeholder(dtype=tf.float32, shape=[None, maxSeqLength, input_dim])     # [batch, 250, 300]
y = tf.placeholder(dtype=tf.int64, shape=[None])    # [batch]
y_onehot = tf.one_hot(indices=y, depth=num_class)   # [batch , 3]
onehot_shape = tf.shape(y_onehot)

# Define LSTM model
lstmCell = [tf.nn.rnn_cell.BasicLSTMCell(num_units=num_unit) for _ in range(3)]
lstmCell = tf.nn.rnn_cell.MultiRNNCell(lstmCell)
lstmCell = tf.nn.rnn_cell.DropoutWrapper(cell=lstmCell, output_keep_prob=drop_out)
val, _ = tf.nn.dynamic_rnn(cell=lstmCell, inputs=x, dtype=tf.float32)   # val: [batch, num_words, num_unit]
val_trans = tf.transpose(val, [1, 0, 2])

output = tf.gather(val_trans, int(val_trans.get_shape()[0]) - 1)

logit = fc(output, num_out=num_class, activation=None)
cost_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=logit)
cost = tf.reduce_mean(cost_entropy)
opt = tf.train.AdadeltaOptimizer(learning_rate=lr).minimize(loss=cost)
y_pred = tf.nn.softmax(logit)
y_pred_cls = tf.argmax(y_pred, dimension=1)


correct = tf.equal(y_pred_cls, y)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

var = [var for var in tf.all_variables()]
saver = tf.train.Saver(var_list=var)

os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.97   # set gpu usage ratio

# logging training progress
with open('./log/log_{}.txt'.format(test_num), 'a') as fp:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        tr_size = len(label_tr)
        iteration = tr_size // batch_size + 1
        print('tr: ', len(label_tr), ' ts: ', len(label_ts))
        fp.write('tr: {}\tts: {}\n'.format(len(label_tr), len(label_ts)))
        print('EPOCH: {}\tBatch Size: {}\tIter:{}'.format(epoch, batch_size, iteration))
        fp.write('EPOCH: {}\tBatch Size: {}\tIter:{}\n\n'.format(epoch, batch_size, iteration))
        for e in range(epoch):
            s_bat = 0
            e_bat = s_bat + batch_size
            cost_t = acc_t = 0
            if e % 15 == 0:
                lr /= 5
            for i in range(iteration):
                X = zero_padding(vec_tr[s_bat:e_bat])
                Y = label_tr[s_bat:e_bat]
                c, _ = sess.run([cost, opt], feed_dict={x: X, y: Y})
                pred, acc = sess.run([y_pred_cls, accuracy], feed_dict={x: X, y: Y})

                cost_t += c / iteration
                acc_t += acc / iteration

                s_bat = e_bat
                if i < iteration - 1:
                    e_bat = s_bat + batch_size
                else:
                    e_bat = None
            X = zero_padding(vec_ts)
            Y = label_ts
            pred, cor, acc = sess.run([y_pred_cls, correct, accuracy], feed_dict={x: X, y: Y})
            print('[{}]\tCOST: {:.5f}\tACC:{:.4f}\t\tACC_TEST:{:.4f}'.format(e, cost_t, acc_t, acc))
            fp.write('[{}]\tCOST: {:.5f}\tACC:{:.4f}\t\tACC_TEST:{:.4f}\n'.format(e, cost_t, acc_t, acc))
            if e % 50 == 0:
                with open('./result/result{}-{}({:.2f}).txt'.format(test_num, e, acc), mode='w') as f:
                    for idx, sentence in enumerate(sent_ts):
                        if cor[idx]:
                            f.write('{} @{} @{}\n'.format(sentence, label_map[pred[idx]], label_map[label_ts[idx]]))

            # save check point
            saver.save(sess=sess, save_path='./weight-{}_{}-{:.3f}.ckpt'.format(test_num, e, acc))
