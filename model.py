import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_decode,crf_log_likelihood
from BatchGenerator import BatchGenerator
from corpus.utils import test_input
import pickle
import sys
with open('BosonNLPtmp.pkl', 'rb') as inp:
	word2id = pickle.load(inp)
	id2word = pickle.load(inp)
	tag2id = pickle.load(inp)
	id2tag = pickle.load(inp)
	x_train = pickle.load(inp)
	y_train = pickle.load(inp)
	x_test = pickle.load(inp)
	y_test = pickle.load(inp)

epochs = 31
batch_size = 32


print ("train len:",len(x_train))
print ("test len:",len(x_test))
print ("word2id len", len(word2id))
print ('Creating the data generator ...')
data_train = BatchGenerator(x_train, y_train, shuffle=True)
# data_valid = BatchGenerator(x_valid, y_valid, shuffle=False)
data_test = BatchGenerator(x_test, y_test, shuffle=False)
print ('Finished creating the data generator.')

config = {}
config["lr"] = 0.001
config["emb_dim"] = 100
config["sen_len"] = len(x_train[0])
config["batch_size"] = batch_size
config["emb_size"] = len(word2id)+1
config["tag_size"] = len(tag2id)

class Model(object):
    """BILSTM-CRF Model"""
    def __init__(self, config, dropout_keep=1):
        super(Model, self).__init__()
        self.lr = config['lr']  #0.001
        self.batch_size = config['batch_size']  # 32
        self.emb_size = config['emb_size']  # 3448
        self.emb_dim = config['emb_dim']  # 100
        self.sen_len = config['sen_len']  # 60
        self.tag_size = config['tag_size']  # 23
        self.dropout_keep = dropout_keep
        self.input_data = tf.placeholder(tf.int32, shape=[self.batch_size, self.sen_len], name='input')  # 32,60
        self.targets = tf.placeholder(tf.int32, shape=[self.batch_size, self.sen_len], name='target')  # 32,60
        with tf.variable_scope('bilstm_crf') as scope:
            self._build_net()

    def _build_net(self):
        word_emb = tf.get_variable('word_emb', [self.emb_size, self.emb_dim])  # 3448*100
        input_emb = tf.nn.embedding_lookup(word_emb, self.input_data)  # 32*60*100
        input_emb = tf.nn.dropout(input_emb, self.dropout_keep)

        fw_cell = tf.nn.rnn_cell.LSTMCell(self.emb_dim, forget_bias=1.0, state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.LSTMCell(self.emb_dim, forget_bias=1.0, state_is_tuple=True)

        # [batch_size, max_time, cell_fw.output_size]  32*60*3448,  32*3448
        (fw_out, bw_out),states = tf.nn.bidirectional_dynamic_rnn(fw_cell,
                                                           bw_cell,
                                                           input_emb,
                                                           dtype=tf.float32,
                                                           time_major=False,
                                                           scope=None)
        # [2, 32] and [2, 2]
        bilstm_out = tf.concat([fw_out, bw_out], axis=2)
        # 32,2*200,10
        W = tf.get_variable(name='W', shape=[self.batch_size, 2 * self.emb_dim, self.tag_size])
        # 32,10,10
        b = tf.get_variable(name='b', shape=[self.batch_size, self.sen_len, self.tag_size], dtype=tf.float32)

        crf_out = tf.tanh(tf.matmul(bilstm_out, W) + b)
        leng = tf.tile(np.array([self.sen_len]), np.array([self.batch_size]))
        log_likelihood, self.transition_params = crf_log_likelihood(crf_out, self.targets, leng)
        loss = tf.reduce_mean(-log_likelihood)

        self.viterbi_sequence, score = crf_decode(crf_out, self.transition_params, leng)

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        self.saver = tf.train.Saver()

    def train(self, epoces, data_train, batch_size=batch_size,):
        batch_num = int(data_train.y.shape[0] / batch_size)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epoces):
                print("epoch:"+str(epoch))
                print("batch_num:" + str(batch_num))
                for batch in range(batch_num):
                    x_batch, y_batch = data_train.next_batch(batch_size)
                    feed_dict = {self.input_data: x_batch, self.targets: y_batch}
                    pre, _ = sess.run([self.viterbi_sequence, self.train_op], feed_dict=feed_dict)
                    acc = 0
                    if batch % 200 == 0:
                        print("epoch:"+str(epoch)+" -- batch:"+str(batch)+" -- batch_len:"+str(len(y_batch)))
                        for i in range(len(y_batch)):
                            for j in range(len(y_batch[0])):
                                if y_batch[i][j] == pre[i][j]:
                                    acc += 1
                        print("epoch:"+str(epoch)+" -- acc_rate:"+str(float(acc) / (len(y_batch) * len(y_batch[0]))))
                if epoch % 3 == 0:
                    path_name = "./model/model" + str(epoch) + ".ckpt"
                    print("save", path_name)
                    self.saver.save(sess, path_name)
    def predict(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state("./model")
            if ckpt is None:
                print("ckpt is None")
                return
            path = ckpt.model_checkpoint_path
            self.saver.restore(sess,path)
            test_input(self,sess,word2id,id2tag,batch_size)

    pass

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("usage:python model.py [train|predict]")
    if len(sys.argv) == 2:
        if sys.argv[1] == 'train':
            model = Model(config)
            model.train(epochs,data_train)
        if sys.argv[1] == 'predict':
            model = Model(config)
            model.predict()


