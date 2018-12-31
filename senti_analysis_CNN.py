import numpy as np
import os
import pickle
import argparse
import tensorflow as tf
from tensorflow.contrib import slim

"""
2018.12.20
Sentiment analysis using CNN
"""
parse = argparse.ArgumentParser()
parse.add_argument('--lr', type=float, default=0.005, help='the learning rate')
parse.add_argument('--epoch', type=int, default=2000, help='the number of epochs')
parse.add_argument('--batch', type=int, default=256, help='the batch size')
parse.add_argument('--gpu', type=int, default=0, help='Select gpu number. Default is 0')
parse.add_argument('--testnum', type=str, default='e-35_256_2000_1', help='Prefix of test name')

label_map = ['negative', 'neutral', 'positive']
args = parse.parse_args()
data_dic = ''
data_file = ''      # define path to pre-embedded dataset
data_path = os.path.join(data_dic, data_file)
test_num = args.testnum
gpu = args.gpu
batch_size = args.batch
epoch = args.epoch
lr = args.lr

input_dim = 300
maxSeqLength = 30
num_class = 3
drop_out = 0.5
win_size = [2, 3, 4]


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

print('Tr / Ts: {} / {}'.format(len(label_tr), len(label_ts)))
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


def conv2d(x, size, n_filter, stride):
    conv = slim.conv2d(inputs=x, num_outputs=n_filter, kernel_size=size, stride=stride,
                       weights_initializer=slim.xavier_initializer_conv2d(), padding='VALID')

    pool = slim.max_pool2d(inputs=conv, kernel_size=(conv.shape[1], conv.shape[2]))
    transpose = tf.transpose(a=pool, perm=[0, 3, 2, 1])
    return transpose


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
    input_mat = np.expand_dims(input_mat, axis=4)
    return input_mat


x = tf.placeholder(dtype=tf.float32, shape=[None, maxSeqLength, input_dim, 1])  # [batch, maxSeq, 300, 1]
y = tf.placeholder(dtype=tf.int64, shape=[None])    # [batch]
y_onehot = tf.one_hot(indices=y, depth=num_class)   # [batch , 3]

conv1 = conv2d(x=x, size=[win_size[0], input_dim], n_filter=2, stride=1)
conv2 = conv2d(x=x, size=[win_size[1], input_dim], n_filter=2, stride=1)
conv3 = conv2d(x=x, size=[win_size[2], input_dim], n_filter=2, stride=1)

fc_feat = tf.concat(values=[conv1, conv2, conv3], axis=1)
fc_feat = tf.squeeze(fc_feat, axis=[2, 3])
logit = fc(fc_feat, num_class, activation=None)

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
with open('./log_{}.txt'.format(test_num), 'a') as fp:
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        tr_size = len(label_tr)
        ts_size = len(label_ts)
        iteration = tr_size // batch_size + 1
        print('tr: ', len(label_tr), ' ts: ', len(label_ts))
        print('EPOCH: {}\tBatch Size: {}\tIter:{}'.format(epoch, batch_size, iteration))
        for e in range(epoch):
            s_bat = 0
            e_bat = s_bat + batch_size
            cost_t = acc_t = 0
            if e % 20 == 0:
                lr /= 10
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

            # test every epochs
            iteration_ts = ts_size // batch_size + 1
            ts_bat = 0
            te_bat = ts_bat + batch_size
            acc_ts = 0
            for i in range(iteration_ts):
                X = zero_padding(vec_ts[ts_bat:te_bat])
                Y = label_ts[ts_bat:te_bat]
                pred, ans, cor, acc_ = sess.run([y_pred_cls, y, correct, accuracy], feed_dict={x: X, y: Y})
                acc_ts += acc_ / iteration_ts

                # every 200 epochs, record test result
                if e % 200 == 0:
                    with open('./result/result{}-{}.txt'.format(test_num, e), mode='w') as f:
                        for idx, sentence in enumerate(sent_ts[ts_bat:te_bat]):
                            if cor[idx]:
                                f.write('{} @{} @{}\n'.format(sentence, label_map[pred[idx]], label_map[Y[idx]]))

                    os.makedirs('./weights/{}/', exist_ok=True)
                    saver.save(sess=sess, save_path='./weights/{}/weight-{}-{:.3f}.ckpt'.format(test_num, e, acc_))

                ts_bat = te_bat
                if i < iteration - 1:
                    te_bat = ts_bat + batch_size
                else:
                    te_bat = None

            print('[{}]\tCOST: {:.5f}\tACC:{:.4f}\t\tACC_TEST:{:.4f}'.format(e, cost_t, acc_t, acc_ts))
            fp.write('[{}]\tCOST: {:.5f}\tACC:{:.4f}\t\tACC_TEST:{:.4f}\n'.format(e, cost_t, acc_t, acc_ts))