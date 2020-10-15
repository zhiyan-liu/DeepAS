import numpy as np
import tensorflow as tf
from six.moves import cPickle
from math import log
from tensorflow.keras.layers import Conv2D, Flatten, MaxPool2D
import random
import time
import sys

SNR_dB = float(sys.argv[1])

Nt = 4
Nr = 64
Ns = 4
groupSize = 16
groupNum = 4

SNR = 10**(SNR_dB/10)
gamma = SNR/Ns

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def slice2input(channel):
    channel = channel.reshape([Nr,Nt])
    channel = np.stack([np.real(channel),np.imag(channel),np.abs(channel)],axis=2)
    return channel

def prepare_data():
    with open('DatasetUlaNormForDeploy.pkl','rb') as cPickle_file:
        channel_list, test_list = cPickle.load(cPickle_file)
    channel_list = [slice2input(channel) for channel in channel_list]
    channel_list = np.asarray(channel_list,dtype = np.float32)
    print(channel_list.shape)
    dataset = tf.data.Dataset.from_tensor_slices(channel_list)
    dataset = dataset.shuffle(300000).batch(128)
    test_list = [slice2input(channel) for channel in test_list]
    return dataset,test_list

def convert_to_dataset(channel_list):
    channel_list = np.asarray(channel_list,dtype = np.float32)
    test_data = tf.convert_to_tensor(channel_list)
    return channel_list

def calcCapacity(H,delta,SNR):
    M = H.shape[0]
    return log(abs(np.linalg.det(np.eye(M,dtype = np.complex)+np.dot(np.dot(delta,H),H.conj().T)*SNR/Ns)),2)

def eval_model(model,test_list):
    totalCapacity = 0
    for i in range(1000):
        test_batch = test_list[50*i:50*(i+1)]
        test_data = convert_to_dataset(test_batch)
        delta,_ = model(test_data,training = False)
        delta = delta.numpy()
        for n in range(delta.shape[0]):
            diag = np.diag(delta[n,:,:])
            groupList = [range(i,Nr,groupNum) for i in range(groupNum)]
            discDelta = np.zeros([Nr,Nr],dtype = np.complex)
            for group in groupList:
                ind = np.argmax(diag[group])
                discDelta[list(group)[ind],list(group)[ind]] = 1
            H = test_batch[n][:,:,0] + 1j * test_batch[n][:,:,1]
            totalCapacity = totalCapacity + calcCapacity(H,discDelta,SNR)
    return totalCapacity / len(test_list)

def weight_variable(shape):
    initializer = tf.initializers.GlorotNormal()
    initial = initializer(shape,dtype=tf.dtypes.float32)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0, shape=shape, dtype=tf.dtypes.float32)
    return tf.Variable(initial)

def log2(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator

def write_pkl(data,filename):
    with open(filename,'wb') as cPickle_file:
        cPickle.dump(data,cPickle_file,protocol = cPickle.HIGHEST_PROTOCOL)

def hardmax(logits):
    hardMax = tf.one_hot(tf.math.argmax(logits, axis = 1), depth = groupSize)
    softMax = tf.nn.softmax(logits)
    return tf.stop_gradient(hardMax - softMax) + softMax

def gumbel_softmax(logits, temp):
    BS = tf.shape(logits)[0]
    num_cat = tf.shape(logits)[1]
    GN = -tf.math.log(-tf.math.log(tf.random.uniform((BS,num_cat),0,1)+1e-20)+1e-20) 
    return tf.nn.softmax((logits + GN) / temp)

def st_gumbel(logits, temp):
    BS = tf.shape(logits)[0]
    num_cat = tf.shape(logits)[1]
    GN = -tf.math.log(-tf.math.log(tf.random.uniform((BS,num_cat),0,1)+1e-20)+1e-20)
    perturbedLog = GN + logits
    hardSamples = tf.one_hot(tf.math.argmax(perturbedLog, axis = 1), depth = groupSize)
    softSamples = tf.nn.softmax(perturbedLog / temp)
    return tf.stop_gradient(hardSamples - softSamples) + softSamples

class Model:
    def __init__(self):
        self.totalSteps = 150000//128 * 15
        self.tEnd = 0.04
        self.alpha = -log(self.tEnd)/self.totalSteps
        self.convFilter1 = weight_variable([3,3,3,64])
        self.convBiase1 = bias_variable([64,4,64])
        self.convFilter2 = weight_variable([3,3,64,64])
        self.convBiase2 = bias_variable([64,4,64])
        self.convFilter3 = weight_variable([3,3,64,64])
        self.convBiase3 = bias_variable([64,4,64])
        self.convFilter4 = weight_variable([3,3,64,64])
        self.convBiase4 = bias_variable([64,4,64])
        self.convFilter5 = weight_variable([3,3,64,64])
        self.convBiase5 = bias_variable([64,4,64])
        # self.convFilter6 = weight_variable([3,3,64,64])
        # self.convBiase6 = bias_variable([64,4,64])
        self.flatten = Flatten()
        self.W_fc1 = weight_variable([16384//4,512])
        self.b_fc1 = bias_variable([512])
        self.W_fc2 = weight_variable([512,512])
        self.b_fc2 = bias_variable([512])
        self.W_phi = [weight_variable([512,groupSize]) for i in range(groupNum)]
        self.bn_layers = [tf.keras.layers.BatchNormalization() for i in range(8)]
        self.dropout = tf.keras.layers.Dropout(.2)
        self.b_phi = [bias_variable([groupSize]) for i in range(groupNum)]
        self.temp = 1
        self.step = 0
        self.iter = 0
        self.MaxPooling =MaxPool2D(pool_size=(2,2))
    def trainable_variables(self):
        return [self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2,\
        self.convFilter1, self.convBiase1, self.convFilter2, self.convBiase2,\
        self.convFilter3, self.convBiase3, self.convFilter4, self.convBiase4,\
        self.convFilter5, self.convBiase5, self.convFilter6, self.convBiase6] + \
        self.W_phi + self.b_phi
    def __call__(self, x, training = True, h = None, use_hardmax = False):
        batch_size = x.shape[0]
        if h == None:
            h = x
        h_mat = tf.complex(h[:,:,:,0],h[:,:,:,1])
        h_mat_conj = tf.complex(h[:,:,:,0],-h[:,:,:,1])
        x = tf.nn.conv2d(x,self.convFilter1,1,"SAME") + self.convBiase1
        x = tf.nn.relu(self.bn_layers[0](x,training = training)) 
        x_res = tf.nn.conv2d(x,self.convFilter2,1,"SAME") + self.convBiase2
        x_res = tf.nn.relu(self.bn_layers[1](x_res,training = training))
        x_res = tf.nn.conv2d(x_res,self.convFilter3,1,"SAME") + self.convBiase3
        x = tf.nn.relu(self.bn_layers[2](x_res,training = training)) + x
        x_res = tf.nn.conv2d(x,self.convFilter4,1,"SAME") + self.convBiase4
        x_res = tf.nn.relu(self.bn_layers[3](x_res,training = training)) 
        x_res = tf.nn.conv2d(x_res,self.convFilter5,1,"SAME") + self.convBiase5
        x = tf.nn.relu(self.bn_layers[4](x_res,training = training)) + x
        x = self.MaxPooling(x)
        x = self.flatten(x)
        #x = self.dropout(x, training = training)
        h_fc1 = tf.nn.relu(tf.matmul(self.bn_layers[6](x,training), self.W_fc1) + self.b_fc1)
        #h_fc1 = self.dropout(h_fc1, training = training)
        h_fc2 = tf.nn.relu(tf.matmul(self.bn_layers[7](h_fc1,training), self.W_fc2) + self.b_fc2)
        #h_fc2 = self.dropout(h_fc2, training = training)
        if not use_hardmax:
            groupList = [tf.nn.softmax((tf.matmul(h_fc2,self.W_phi[i])+self.b_phi[i])) for i in range(groupNum)]
        else:
            if self.iter > 1000:
                groupList = [st_gumbel(tf.matmul(h_fc2,self.W_phi[i]+self.b_phi[i]),self.temp) for i in range(groupNum)]
            else:
                groupList = [gumbel_softmax(tf.matmul(h_fc2,self.W_phi[i]+self.b_phi[i]),self.temp) for i in range(groupNum)]
        diag = tf.stack([groupList[int(i%groupNum)][:,int(i/groupNum)] for i in range(Nr)],axis = 1)
        delta = tf.linalg.diag(diag)
        #ent = -tf.reduce_sum(tf.math.multiply_no_nan(log2(diag),diag))
        delta_comp = tf.complex(delta,tf.zeros(shape = [batch_size,Nr,Nr],dtype = tf.dtypes.float32))
        capacity = -tf.reduce_sum(log2(tf.abs(tf.linalg.det(tf.eye(Nr, dtype=tf.dtypes.complex64)+gamma*tf.matmul(delta_comp,tf.matmul(h_mat,h_mat_conj,transpose_b = True))))))/batch_size
        return delta,capacity
    def update_temp(self):
        if self.temp > 0.3:
            step = 0.1
        else:
            step = 0.03
        self.temp = max([0.1, self.temp - step])
        #self.temp = max([0.5, 3*np.exp(-iter*np.log(6)/19)])
        self.iter += 1

    def update_step(self):
        self.temp = max([self.tEnd, np.exp(- self.step * self.alpha)])
        self.step += 1
        

dataset,test_data = prepare_data()

#PRETRAINING WITH CLEAN DATA

optimizer = tf.keras.optimizers.Adam()
delta_history = []
loss_history = []
loss_by_epoch = []
testcap_history = []
max_testcap = 0.0
model = Model()
print(eval_model(model,test_data))
niter = 30

for epochs in range(niter):
    history_start = len(loss_history)
    for (batch, x) in enumerate(dataset):
        with tf.GradientTape() as tape:
            delta,capacity = model(x, use_hardmax = True)
            loss_history.append(capacity.numpy().mean())
            trainable_variables = model.trainable_variables()
            grads = tape.gradient(capacity, trainable_variables)
            iter = optimizer.apply_gradients(zip(grads, trainable_variables))
            model.update_step()
    history_end = len(loss_history)
    loss_by_epoch.append(np.mean(loss_history[history_start:history_end]))
    print('Epoch '+str(epochs)+', loss = '+str(np.mean(loss_history[history_start:history_end])))
    test_cap = eval_model(model,test_data)
    testcap_history.append(test_cap)
#    model.update_temp()
    print('Achieved capacity on test set = ' + str(test_cap))
    print('Temp = ' + str(model.temp))
    if test_cap > max_testcap:
        write_pkl(model,'best_as_model_nonst_SNR_res' + str(SNR_dB) + '.pkl')
        print('New best model, model saved.')
        max_testcap = test_cap

# with open('best_as_model_nonst.pkl','rb') as cPickle_file:
#     bestModel = cPickle.load(cPickle_file)

# #TEST WITH PREDICTED DATA
# with open('test_data.pkl','rb') as cPickle_file:
#     while True:
#         try:
#             h_pred_test, h_real_test = cPickle.load(cPickle_file)
#         except EOFError:
#             break

# h_pred_test = h_pred_test[0:1000]
# h_real_test = h_real_test[0:1000]
# h_pred_test = [slice2input(channel) for channel in h_pred_test]
# h_real_test = [slice2input(channel) for channel in h_real_test]
# test_data = convert_to_dataset(h_pred_test)

# start_time = time.time()
# delta,_ = bestModel(test_data,training = False)
# delta = delta.numpy()
# totalCapacity = 0
# groupList = [range(i,Nr,groupNum) for i in range(groupNum)]
# for n in range(delta.shape[0]):
#     diag = np.diag(delta[n,:,:])
#     discDelta = np.zeros([Nr,Nr],dtype = np.complex)
#     for group in groupList:
#         ind = np.argmax(diag[group])
#         discDelta[list(group)[ind],list(group)[ind]] = 1
#     H = h_real_test[n][:,:,0] + 1j * h_real_test[n][:,:,1]
#     totalCapacity = totalCapacity + calcCapacity(H,discDelta,SNR)

# end_time = time.time()
# cnn_time = end_time - start_time
# print(totalCapacity / delta.shape[0])
# print(cnn_time)
