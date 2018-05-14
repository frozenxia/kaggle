import tensorflow as tf
import numpy as np
import pandas as pd
# from sklearn.preprocessing import OneHotEncoder
import tensorflow.contrib.keras  as kr

from tqdm import tqdm
def get_train_data():
    train = pd.read_csv('data/train.csv',sep=',')
    train= train.iloc[np.random.permutation(train.shape[0])]
    train_x = train.iloc[:,1:].as_matrix()
    ## need one-hot encode
    train_y = train.iloc[:,:1].as_matrix()

    # enc = OneHotEncoder()

    train_y = kr.utils.to_categorical(train_y,10)
    return train_x,train_y

def get_batch(train_x,train_y,batch_size):
    lenx = len(train_x)
    iter_n = int((lenx-1)/batch_size)+1
    for i in range(iter_n):
        yield train_x[i*batch_size:(i+1)*batch_size],train_y[i*batch_size:(i+1)*batch_size]



def model():
    input_x = tf.placeholder(tf.float32,name='input_x',shape=[None,784])
    input_y = tf.placeholder(tf.float32,name='input_y',shape=[None,10])
    dropt_out = tf.placeholder(tf.float32,name='drop_out')


    input_x_reshape = tf.reshape(input_x,shape=[-1,28,28],name='input_x_reshape')
    input_x_reshape = tf.expand_dims(input_x_reshape, -1)

    ## cnn

    def get_cnn_out(i):
        with tf.variable_scope('cnn_filter_%d'%(i)):
            cnn_filter = tf.get_variable('cnn_filter_%d'%(i), [i, i, 1, 64], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
            cnn_out = tf.nn.conv2d(input_x_reshape,cnn_filter,strides=[1,1,1,1],padding='VALID')
            # add max_pooling
            cnn_out_pooling = tf.nn.max_pool(cnn_out,[1,i,i,1],[1,1,1,1],padding='VALID')
            shape_size = (28-i+1 -i+1)*(28-i+1-i+1)
            return tf.reshape(cnn_out_pooling,shape=[-1,shape_size,64])
    cnn_out3 = get_cnn_out(3)
    # print(cnn_out3)
    cnn_out4 = get_cnn_out(4)
    cnn_out5 = get_cnn_out(5)

    cnn_out= tf.concat([cnn_out3,cnn_out4,cnn_out5],1)

    ## lstm
    rnn_layers = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(size),input_keep_prob=dropt_out) for size in [128, 256]]
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    lstm_outputs,final_state = tf.nn.dynamic_rnn(multi_rnn_cell,cnn_out,dtype=tf.float32)


    ## dense
    dout1 = tf.layers.dense(lstm_outputs[:,-1,:],128)
    dout2 = tf.layers.dense(dout1,10)
    # print(dout2)

    ## accuracy and logits
    predict = tf.nn.softmax(dout2)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=dout2,labels=input_y)
    loss = tf.reduce_mean(cross_entropy)
    adamp = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(predict,1),tf.arg_max(input_y,1)),tf.float32))
    return input_x,input_y,predict,loss,accuracy,adamp,dropt_out



def run():
    origin_x,origin_y = get_train_data()
    isplit = int(len(origin_x)*0.8)
    # train
    train_x = origin_x[:isplit]
    train_y = origin_y[:isplit]

    # validation

    valid_x =  origin_x[isplit:]
    valid_y =  origin_y[isplit:]
    input_x,input_y,predict,loss,accuracy,adamp,dropt_out = model()
    count = 0
    with tf.Session() as session:
        session.run(tf.initialize_all_variables())

        def validation():

            loss_t = 0.0
            acc_t = 0.0
            count_t = 0
            for bx,by in get_batch(valid_x,valid_y,128):
                feed={
                    input_x:bx,
                    input_y:by,
                    dropt_out:1
                }
                lb = len(bx)
                count_t += lb
                vloss,vacc = session.run([loss,accuracy],feed_dict=feed)
                loss_t += vloss*lb
                acc_t = lb*vacc

            return acc_t/count_t,loss_t/count_t

        for i in range(100):
            
            for bx,by in get_batch(train_x,train_y,64):
                # print(by)
                feed={
                    input_x:bx,
                    input_y:by,
                    dropt_out:0.5
                }
                vloss,_ = session.run([loss,adamp],feed_dict=feed)
                print('loss : ',vloss)
                count +=1
                if count %100 == 0:
                    print('validation : %f,%f'%(validation()))



run()
