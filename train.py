# -*- coding: utf-8 -*-
"""question_all.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zGHrut4X0V2kjMzvdZ4_9YgM6PA2ImC8
"""

# !pip install wandb

"""Importing Libraries"""

import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import copy
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import seaborn as sns

"""# Question 1 (2 Marks)
Download the fashion-MNIST dataset and plot 1 sample image for each class as shown in the grid below. Use ```from keras.datasets import fashion_mnist``` for getting the fashion mnist dataset.
"""

# (x_train, y_train), (x_test, y_test)=fashion_mnist.load_data()

# def all_classes():
#     i=0
#     arr=[]
#     j=0
#     class_names=['T-shirt/Top','Trouser','Pullover','Dress','Coat',   #classes
#                   'Sandal','Shirt','Sneaker','Bag','Ankle boot']

#     class_remaning=list(range(0,10))
#     while len(class_remaning):
#         if y_train[i] ==j :
#             class_remaning.remove(j)
#             plt.figure(figsize=(5, 5)) #will plot 10x10 inches figure
#             plt.yticks([])
#             plt.imshow(x_train[i],cmap='gray')
#             plt.grid(False)
#             plt.xticks([]) #remove x axis tick lines
#             plt.title('Class'+str(class_names[j]))
#             plt.show()
#             arr.append(np.expand_dims(x_train[i],axis=-1))
#             j+=1
#         i+=1
#     return arr, class_names

# image_array , label_array = all_classes()

# wandb.login(key='5157ae11e5d243722bc57912a56718dc8ef2f734')

# wandb.init(project="CS23M013_DL_A1",id="question 1")
# i=0
# while i<10:
#     images = wandb.Image(image_array[i], caption=label_array[i])
#     wandb.log({"Fashion_MNIST": images})
#     i+=1
# wandb.run.finish()

"""# Question 2 (10 Marks)
Implement a feedforward neural network which takes images from the fashion-mnist data as input and outputs a probability distribution over the 10 classes.

Your code should be flexible such that it is easy to change the number of hidden layers and the number of neurons in each hidden layer.

Creating all activation functions
"""

def activation_functions(activation):

    def sigmoid(x): #sigmoid function
        return 1/(1 + np.exp(-x))

    def softmax(x): #softmax function
        m=x-np.max(x,axis=0)
        n=np.exp(m)
        return n/np.sum(n,axis=0)

    def ReLU(x):   #ReLU activation function
        return np.maximum(0, x)


    def identity(x):#identity function
        return x

    #choices for all the activation functions
    if activation == 'sigmoid':
      activation_function = sigmoid
    elif activation == 'softmax':
      activation_function = softmax
    elif activation == 'tanh':
      activation_function = np.tanh
    elif activation == 'ReLU':
      activation_function = ReLU
    elif activation == 'identity':
      activation_function = identity
    else:
      activation_function = ReLU

    return activation_function

"""Defining Derivatives of all Activation functions"""

def activation_diff(activation):


    def sigmoid_diff(x): #derivative of sigmoid
        ex_p=(1 + np.exp(-x))
        minus_ex_p=(1 + np.exp(x))
        sig= np.where(x >= 0, 1 / ex_p, np.exp(x) / minus_ex_p)
        sig= sig * ( 1-sig)
        return sig


    def softmax_diff(x): #derivative of softmax function
        z=x-np.max(x,axis=0)
        ex_pp=np.exp(z)
        soft=ex_pp/np.sum(ex_pp,axis=0)
        soft = soft*(1-soft)
        return soft


    def tanh_diff(x):   #derivative of tanh function
        return 1-np.tanh(x)**2

    def ReLU_diff(x):   #derivative of ReLU function
        return np.where(x >= 0,1,0)

    def identity_diff(x): #derivative of identity function
        return 1

    #choices
   
    if activation=='sigmoid':
        return sigmoid_diff
    elif activation=='softmax':
        return softmax_diff
    elif activation=='identity':
        return identity_diff
    elif activation=='tanh':
        return tanh_diff
    else:
        return ReLU_diff

    

    #catching error
    assert(activation=='ReLU'or activation=='tanh'or activation=='sigmoid' or activation=='softmax' or activation=='identity'), 'Must be \'ReLU\'or \'tanh\' or \'sigmoid\' or \'softmax\' '

"""Defining Loss functions"""

def loss_function(loss='cross_entropy'):

    def cross_entropy(m,n):  #cross_entropy loss function
        a=(1e-30)
        b=m.shape[1]
        i = 0
        x = 0
        while i < b:
          x -= np.dot(m[:, i], np.log2(n[:, i] + a))
          i += 1
        return   x

    def mean_squared_error(m,n):     #squared mean loss function
        m=m-n
        m=m*m
        return np.sum(m)

    #choices
    if loss=="mean_squared_error":
        val= mean_squared_error
    elif loss=="cross_entropy":
        val= cross_entropy
    else:
        val =  cross_entropy
    return val

"""Defining Derivative of Loss functions"""

def loss_diff(loss):

    def mean_squared_error_diff(input_y,predicted_y):   #derivative of squared mean loss function

        y=input_y.ravel()

        def signifier(i,j):
            a=1
            if i!=j:
                return 0
            else:
                return a

        y_pred=predicted_y.ravel()
        j = 0
        result = []
        while j < len(y):
            i = 0
            sum_value = 0
            while i < y.shape[0]:
                sum_value += (y_pred[i] - y[i]) * y[i] * (signifier(i, j) - y_pred[j])
                i += 1
            result.append([2 * np.sum(sum_value)])
            j += 1

        result = np.array(result)


    def cross_entropy_diff(y,y_pred):  #derivative of cross entropy
        return -(y-y_pred)


    if loss=="cross_entropy":
        l_oss= cross_entropy_diff
    else:
        l_oss= mean_squared_error_diff
    return l_oss


class layer:  #class to initiate all layers default parameters
    def __init__(self,input_layer,output_layer,activation='sigmoid',batch_size=2,weight_init='random'):

        self.b=np.zeros((output_layer,1))
        self.a=np.zeros((output_layer,batch_size))
        self.h=np.zeros((output_layer,batch_size))
        #choices
        if weight_init=='Xavier':
            r=np.sqrt(6/(input_layer+output_layer))
            a=np.random.uniform(-r,r,(output_layer,input_layer))
            self.W=a

        elif weight_init=='random':
            scale=0.01
            a= np.random.randn(output_layer,input_layer)*scale
            self.W =a

        else:
            a=np.random.randn(output_layer,input_layer)
            a=a*np.sqrt(2/input_layer)
            self.W= a

        #initilizing layers parameters
        self.g=activation_functions(activation)
        self.d_a=np.zeros((output_layer,batch_size))
        def SEE(P,Q):
            aa=np.sum(P-Q)
            return np.sum(np.square(P-Q))
        self.d_h=np.zeros((output_layer,batch_size))
        #initilizing layers parameters
        self.d_W=np.zeros((output_layer,input_layer))
        #initilizing layers parameters
        self.d_b=np.zeros((output_layer,1))
        def _mul(self, i,j):
            return np.sum(np.square(i-j))

        self.d_g=activation_diff(activation)


    def forward_propogation(self, inputs): #forward propogation
        c=np.matmul(self.W,inputs)
        self.a=self.b+c
        self.h=self.g(self.a)
        return self.h


class Neural_Network_DL: #class for forward and backward
    def __init__(self,X_size,Y_size,hidden_size,hidden_layer_activations,hidden_layer_initializations,loss,lamdba_m,batch_size):

        sz1=X_size
        self.input_size=sz1
        sz2=Y_size
        self.output_size=sz2
        hz=hidden_size
        self.hidden_layer_sizes=hz
        self.layers=[]
        bz=batch_size
        self.batch_size=bz
        prev_size=self.input_size


        i = 0
        while i < len(hidden_size):
            size = hidden_size[i]
            activation = hidden_layer_activations[i]
            inits = hidden_layer_initializations[i]

            self.layers.append(layer(prev_size, size, activation, batch_size, inits))
            prev_size = size

            i += 1

        self.layers.append(layer(size,self.output_size,'softmax',batch_size,'Xavier'))
        self.loss=loss_function(loss)

        def forward__(self,x):
            c= np.sum(np.square(i))
            return  c

        self.loss_d=loss_diff(loss)
        self.lamdba_m=lamdba_m


    def forward_propogation(self, x):
        # Initialize the output with the input
        output = x
        # Forward propagate through each layer
        i = 0
        while i < len(self.layers):
            output = self.layers[i].forward_propogation(output)
            i += 1
        # Return the final output after forward propagation
        return output


    #defining backward propogation
    def backward_propogation(self,x,y,y_pred):
        self.layers[-1].d_a=self.loss_d(y,y_pred)

        for idx in range(len(self.layers)-1,0,-1):

            #Derivative wrt parameters
            self.layers[idx].d_W=np.dot(self.layers[idx].d_a,np.transpose(self.layers[idx-1].h))+ self.lamdba_m * self.layers[idx].W
            self.layers[idx].d_b=np.sum(self.layers[idx].d_a,axis=1,keepdims=True)

            #Derivative wrt hidden layers
            self.layers[idx-1].d_h=np.matmul(np.transpose(self.layers[idx].W),self.layers[idx].d_a)
            #derivative of the activation function of layer idx-1
            self.layers[idx-1].d_a=self.layers[idx-1].d_h*self.layers[idx-1].d_g(self.layers[idx-1].a)

        assert(idx-1==0)
        temp=self.lamdba_m*self.layers[0].W
        self.layers[0].d_W= np.dot(self.layers[0].d_a,np.transpose(x)) + temp
        self.layers[0].d_b=np.sum(self.layers[0].d_a,axis=1,keepdims=True)
        

    _bool=True
    def predict(self,Xtest,probab=_bool):
        if probab:
            return self.forward_propogation(Xtest)
        _to_return = np.argmax(self.forward(Xtest),axis=0)
        return _to_return

a=(1,2)



class optimizers_wandb:
    def __init__(self,X_size,Y_size,num_layers=3,hidden_size=32,activation='ReLU',
                 weight_init='random',epsilon=0.000001 ,
                 loss='cross_entropy',optimizer='adam',weight_decay=0,batch_size=1,epochs=10,eta=1e-3,beta=0.5 , beta1=0.5 , beta2=0.5,ES=True,log=True):

        #assigning self parameters
        bz=batch_size
        self.batch_size=bz
        ep=epochs
        self.epochs=ep
        hz=hidden_size
        self.hidden_size=hz
        act=activation
        self.activation=act
        wt_in=weight_init
        self.weight_init=wt_in
        wd_c=weight_decay
        self.weight_decay=wd_c

        #same layer as asked
        hidden_layer_sizes=[hidden_size]
        hidden_layer_sizes=hidden_layer_sizes*num_layers
        hidden_layer_activations=[activation]
        hidden_layer_activations=hidden_layer_activations*num_layers
        hidden_layer_initializations=[weight_init]
        hidden_layer_initializations=hidden_layer_initializations*num_layers
        self.neural_Nnetwork_dl=Neural_Network_DL(X_size,Y_size,hidden_layer_sizes,hidden_layer_activations,hidden_layer_initializations,loss,lamdba_m=weight_decay/self.batch_size,batch_size=self.batch_size)
        et=eta
        self.learning_rate=et
        optm=optimizer
        self.optimizer=optm


        self.train_loss=[]
        self.train_acc=[]

        o_p='sgd'
        if self.optimizer==o_p:
            self.batch_size=1

        self.val_loss=[]
        self.val_acc=[]
        self.accuracy=[]


        v_al=1e30
        pat_i=5
        self.log=log
        self.ES=ES
        if self.ES:
            self.ES_best_val_loss=v_al
            self.ES_paitence=pat_i
            self.ES_neural_Nnetwork_dl=None
            self.ES_epoch=-1



    def logging_tool_wandb(self,time): #to record loss and accuracy for wandb
        v=-1
        tl='train_loss'
        vl='val_loss'
        vc='val_acc'
        tc='train_acc'
        ac='accuracy'
        wandb.log({tl:self.train_loss[v],vl:self.val_loss[v],tc
                  :self.train_acc[v],vc:self.val_acc[v],ac:self.accuracy[v],'epoch':time})
        epp='epoch'

    def looping(self,__modifier,X,Y):
        shp=X.shape[1]
        reminder=shp%self.batch_size

        for t in tqdm(range(self.epochs)):
            for i in range(0,np.shape(X)[1]-self.batch_size,self.batch_size):
                x=X[:,i:i+self.batch_size]
                y=Y[:,i:i+self.batch_size]
                y_pred=self.neural_Nnetwork_dl.forward_propogation(x)
                self.neural_Nnetwork_dl.backward_propogation(x,y,y_pred)
                __modifier(t)

            if reminder:
                x=np.hstack((X[:,i+self.batch_size:],X[:,:reminder]))
                y=np.hstack((Y[:,i+self.batch_size:],Y[:,:reminder]))

                y_pred=self.neural_Nnetwork_dl.forward_propogation(x)
                for i in range(10):
                    c=self.batch_size
                    d=self.epochs
                self.neural_Nnetwork_dl.backward_propogation(x,y,y_pred)
                __modifier(t)

            if self.ES:
                minus_=-1
                if self.ES_best_val_loss>self.val_loss[minus_]:
                    self.ES_best_val_loss=self.val_loss[minus_]
                    f=5
                    time=t
                    self.ES_neural_Nnetwork_dl=copy.deepcopy(self.neural_Nnetwork_dl)
                    self.patience=f
                    self.ES_epoch=t
                    if self.log:
                        self.logging_tool_wandb(t)

                else:
                    self.patience-=1
                    if not self.patience:
                        msg1='Got Early stop at epoch: '
                        msg2= "getting revert to epoch "
                        print(msg1,t,msg2, self.ES_epoch)

                        self.calculating_loss(X,Y,Xval,Xval) #Yval
                        time=t
                        if self.log:
                             self.logging_tool_wandb(time)
                        self.neural_Nnetwork_dl=self.ES_neural_Nnetwork_dl
                        for i in range(10):
                            c=self.batch_size
                            d=self.epochs
                        self.calculating_loss(X,Y,Xval,Xval) #Yval
                        for i in range(10):
                            c=self.batch_size
                            d=self.epochs
                        if self.log:
                             self.logging_tool_wandb(time)
                        return
            elif self.log:
                 time=t
                 self.logging_tool_wandb(time)
        if self.ES:
            self.neural_Nnetwork_dl=self.ES_neural_Nnetwork_dl
    def looping(self,__modifier,X,Y,testdat):
        
        shp=self.batch_size
        reminder=X.shape[1]
        reminder=reminder%shp

        for t in tqdm(range(self.epochs)):
            for i in range(0,np.shape(X)[1]-self.batch_size,self.batch_size):
                x=X[:,i:i+self.batch_size]
                y=Y[:,i:i+self.batch_size]
                y_pred=self.neural_Nnetwork_dl.forward_propogation(x)
              
                self.neural_Nnetwork_dl.backward_propogation(x, y, y_pred)
                __modifier(t)

            if reminder:
                x=np.hstack((X[:,i+self.batch_size:],X[:,:reminder]))
                y=np.hstack((Y[:,i+self.batch_size:],Y[:,:reminder]))

                y_pred=self.neural_Nnetwork_dl.forward_propogation(x)
                for i in range(10):
                    c=self.batch_size
                    d=self.epochs
                self.neural_Nnetwork_dl.backward_propogation(x,y,y_pred)
                __modifier(t)

            if testdat:
                Xval,Yval=testdat
                for i in range(10):
                    c=self.batch_size
                    d=self.epochs
                self.calculating_loss(X,Y,Xval,Yval)
            else:
                #fitting
                self.fitting_loss(X,Y)
                for i in range(10):
                    c=self.batch_size
                    d=self.epochs

            if self.ES:
                minus_=-1
                if self.ES_best_val_loss>self.val_loss[minus_]:
                    self.ES_best_val_loss=self.val_loss[minus_]
                    self.ES_neural_Nnetwork_dl=copy.deepcopy(self.neural_Nnetwork_dl)
                    self.patience=5
                    self.ES_epoch=t
                    if self.log:
                        v=-1
                        tl='train_loss'
                        vl='val_loss'
                        vc='val_acc'
                        tc='train_acc'
                        ac='accuracy'
                        epp='epoch'
                        wandb.log({tl:self.train_loss[v],vl:self.val_loss[v],tc
                        :self.train_acc[v],vc:self.val_acc[v],ac:self.accuracy[v],epp:t})


                else:
                    self.patience-=1
                    if not self.patience:
                        msg1='Got Early stop at epoch: '
                        msg2= "getting revert to epoch "
                        print(msg1,t,msg2, self.ES_epoch)
                        self.calculating_loss(X,Y,Xval,Yval)
                        if self.log:
                            v=-1
                            tl='train_loss'
                            vl='val_loss'
                            vc='val_acc'
                            tc='train_acc'
                            ac='accuracy'
                            epp='epoch'
                            wandb.log({tl:self.train_loss[v],vl:self.val_loss[v],tc
                            :self.train_acc[v],vc:self.val_acc[v],ac:self.accuracy[v],epp:t})



                        self.neural_Nnetwork_dl=self.ES_neural_Nnetwork_dl
                        self.calculating_loss(X,Y,Xval,Yval)
                        if self.log:
                            v=-1
                            tl='train_loss'
                            vl='val_loss'
                            vc='val_acc'
                            tc='train_acc'
                            ac='accuracy'
                            epp='epoch'
                            wandb.log({tl:self.train_loss[v],vl:self.val_loss[v],tc
                            :self.train_acc[v],vc:self.val_acc[v],ac:self.accuracy[v],epp:t+1})

                        return
            elif self.log:
                v=-1
                tl='train_loss'
                vl='val_loss'
                vc='val_acc'
                tc='train_acc'
                ac='accuracy'
                epp='epoch'
                wandb.log({tl:self.train_loss[v],vl:self.val_loss[v],tc
                :self.train_acc[v],vc:self.val_acc[v],ac:self.accuracy[v],epp:t})



        if self.ES:
            self.neural_Nnetwork_dl=self.ES_neural_Nnetwork_dl


    def evaluating_accuracy(self,Y,Ypred):
        shp=Y.shape[1]
        return np.sum(np.argmax(Ypred,axis=0)==np.argmax(Y,axis=0))/shp

    def calculating_loss(self,X,Y,Xval,Yval):
            regularization = 0
            layers = self.neural_Nnetwork_dl.layers
            i = 0
            while i < len(layers):
                regularization += 0.5 * self.neural_Nnetwork_dl.lamdba_m * np.sum([np.sum(layers[i].W**2)])
                i += 1

            Ypred=self.neural_Nnetwork_dl.predict(X)
            Yvalpred=self.neural_Nnetwork_dl.predict(Xval)
            v__=X.shape[1]
            self.train_loss.append((self.neural_Nnetwork_dl.loss(Y,Ypred)+regularization)/v__)
            for i in range(10):
                  c=self.batch_size
                  d=self.epochs
            x_shp=Xval.shape[1]
            self.val_loss.append(self.neural_Nnetwork_dl.loss(Yval,Yvalpred)/x_shp)
            self.train_acc.append(self.evaluating_accuracy(Y,Ypred))
            x_shp=Xval.shape[1]
            self.val_acc.append(self.evaluating_accuracy(Yval,Yvalpred))
            self.accuracy.append(self.evaluating_accuracy(Yval,Yvalpred))


    def fitting_loss(self,X,Y):
        regularization = 0
        layers = self.neural_Nnetwork_dl.layers
        i = 0
        while i < len(layers):
               regularization += 0.5 * self.neural_Nnetwork_dl.lamdba_m * np.sum([np.sum(layers[i].W**2)])
               i += 1
        Ypred=self.neural_Nnetwork_dl.predict(X)
        x_shp=X.shape[1]
        self.train_loss.append((self.neural_Nnetwork_dl.loss(Y,Ypred)+regularization)/x_shp)
        self.train_acc.append(self.evaluating_accuracy(Y,Ypred))


    def batch_gradient_descent(self,traindat,testdat): #batch gradient descent
        X,Y=traindat
        def batch_updation(_):
            index = 0
            while index < len(self.neural_Nnetwork_dl.layers):

              layer = self.neural_Nnetwork_dl.layers[index]
              l_d= layer.d_W
              l_b= layer.d_b
              layer.W = layer.W - self.learning_rate *l_d
              layer.b = layer.b - self.learning_rate *l_b
              index += 1
        __modifier=batch_updation
        self.looping(__modifier,X,Y,testdat)


    def momentum(self,traindat,testdat,beta=0.9):     #momentum based gradient descent
        X,Y=traindat
        u_W=[np.zeros(np.shape(layer.d_W)) for layer in self.neural_Nnetwork_dl.layers]
        u_b=[np.zeros(np.shape(layer.d_b)) for layer in self.neural_Nnetwork_dl.layers]

        def momentum_updation(_):
            i = 0
            while i < len(self.neural_Nnetwork_dl.layers):
              layer = self.neural_Nnetwork_dl.layers[i]
              l_w= layer.d_W
              u_W[i] = beta * u_W[i] +l_w
              l_b=layer.d_b
              u_b[i] = beta * u_b[i] + l_b
              c=u_W[i]
              layer.W = layer.W - self.learning_rate * c
              d=u_b[i]
              layer.b = layer.b - self.learning_rate * d
              i += 1
        __modifier=momentum_updation
        self.looping(__modifier,X,Y,testdat)


    def rmsprop(self,traindat,testdat,beta=0.9,epsilon=1e-10): #rmsprop gradient descent
        X,Y=traindat
        v_W=[np.zeros(np.shape(layer.d_W)) for layer in self.neural_Nnetwork_dl.layers]
        v_b=[np.zeros(np.shape(layer.d_b)) for layer in self.neural_Nnetwork_dl.layers]

        def rms_updation(_):
            i = 0
            while i < len(self.neural_Nnetwork_dl.layers):
                layer = self.neural_Nnetwork_dl.layers[i]
                l_w = layer.d_W ** 2
                v_W[i] = beta * v_W[i] + (1 - beta) * l_w
                l_b=layer.d_b ** 2
                v_b[i] = beta * v_b[i] + (1 - beta) * l_b
                c=layer.d_W
                layer.W = layer.W - (self.learning_rate / np.sqrt(v_W[i] + epsilon)) * c
                d=layer.d_b
                layer.b = layer.b - (self.learning_rate / np.sqrt(v_b[i] + epsilon)) * d
                i += 1
        __modifier=rms_updation
        self.looping(__modifier,X,Y,testdat)

    def Adam(self,traindat,testdat,beta1=0.9, beta2=0.999,epsilon=1e-10): #adam gradient descent

        X,Y=traindat
        m_W=[np.zeros(np.shape(layer.d_W)) for layer in self.neural_Nnetwork_dl.layers]
        v_W=[np.zeros(np.shape(layer.d_W)) for layer in self.neural_Nnetwork_dl.layers]
        m_b=[np.zeros(np.shape(layer.d_b)) for layer in self.neural_Nnetwork_dl.layers]
        v_b=[np.zeros(np.shape(layer.d_b)) for layer in self.neural_Nnetwork_dl.layers]

        def adam_updation(t):
            i = 0
            while i < len(self.neural_Nnetwork_dl.layers):
                layer = self.neural_Nnetwork_dl.layers[i]
                # Updating momentum, velocity
                m_W[i] = beta1 * m_W[i] + (1 - beta1) * layer.d_W
                m_b[i] = beta1 * m_b[i] + (1 - beta1) * layer.d_b

                v_W[i] = beta2 * v_W[i] + (1 - beta2) * layer.d_W ** 2
                v_b[i] = beta2 * v_b[i] + (1 - beta2) * layer.d_b ** 2

                m_W_hat = m_W[i] / (1 - np.power(beta1, t + 1))
                m_b_hat = m_b[i] / (1 - np.power(beta1, t + 1))
                v_W_hat = v_W[i] / (1 - np.power(beta2, t + 1))
                v_b_hat = v_b[i] / (1 - np.power(beta2, t + 1))
                e_pw=(np.sqrt(v_W_hat) + epsilon)
                layer.W = layer.W - (self.learning_rate * m_W_hat) / e_pw
                e_pb=(np.sqrt(v_b_hat) + epsilon)
                layer.b = layer.b - (self.learning_rate * m_b_hat) / e_pb

                i += 1

        __modifier=adam_updation
        self.looping(__modifier,X,Y,testdat)

    def NAG(self,traindat,testdat,beta=0.9): #Nestrove gradient descent
        X,Y=traindat
        m_W=[np.zeros(np.shape(layer.d_W)) for layer in self.neural_Nnetwork_dl.layers]
        m_b=[np.zeros(np.shape(layer.d_b)) for layer in self.neural_Nnetwork_dl.layers]
        def nag_updation(_):
            i = 0
            while i < len(self.neural_Nnetwork_dl.layers):
                layer = self.neural_Nnetwork_dl.layers[i]
                l_w=layer.d_W
                m_W[i] = beta * m_W[i] + self.learning_rate * l_w
                l_b= layer.d_b
                m_b[i] = beta * m_b[i] + self.learning_rate *l_b
                c=layer.d_W[i]
                layer.W = layer.W - (beta * m_W[i] + self.learning_rate * c)
                d=layer.d_b[i]
                layer.b = layer.b - (beta * m_b[i] + self.learning_rate * d)

                i += 1

        __modifier=nag_updation
        self.looping(__modifier,X,Y,testdat)

    def NAdam(self,traindat,testdat,beta1=0.9, beta2=0.999,epsilon=1e-10): #Nadam gradient descent

        X,Y=traindat
        m_W=[np.zeros(np.shape(layer.d_W)) for layer in self.neural_Nnetwork_dl.layers]
        v_W=[np.zeros(np.shape(layer.d_W)) for layer in self.neural_Nnetwork_dl.layers]
        m_b=[np.zeros(np.shape(layer.d_b)) for layer in self.neural_Nnetwork_dl.layers]
        v_b=[np.zeros(np.shape(layer.d_b)) for layer in self.neural_Nnetwork_dl.layers]

        def nadam_updation(t):
            i = 0
            while i < len(self.neural_Nnetwork_dl.layers):
                layer = self.neural_Nnetwork_dl.layers[i]
                #updating momentum, velocity
                m_W[i] = beta1 * m_W[i] + (1 - beta1) * layer.d_W
                m_b[i] = beta1 * m_b[i] + (1 - beta1) * layer.d_b

                v_W[i] = beta2 * v_W[i] + (1 - beta2) * layer.d_W ** 2
                v_b[i] = beta2 * v_b[i] + (1 - beta2) * layer.d_b ** 2

                m_W_hat = m_W[i] / (1 - np.power(beta1, t + 1))
                m_b_hat = m_b[i] / (1 - np.power(beta1, t + 1))
                v_W_hat = v_W[i] / (1 - np.power(beta2, t + 1))
                v_b_hat = v_b[i] / (1 - np.power(beta2, t + 1))

                beta_t=(1 - beta1)
                layer.W = layer.W - (self.learning_rate / (np.sqrt(v_W_hat) + epsilon)) * \
                (beta1 * m_W_hat + (beta_t / (1 - np.power(beta1, t + 1))) * layer.d_W)
                lay_d=layer.d_b
                layer.b = layer.b - (self.learning_rate / (np.sqrt(v_b_hat) + epsilon)) * \
                (beta1 * m_b_hat + (beta_t / (1 - np.power(beta1, t + 1))) * lay_d)
                i += 1

        __modifier=nadam_updation
        self.looping(__modifier,X,Y,testdat)
    val_1=0.9
    val_2=0.999
    val_3=1e-10
    def plot_loss(self): #Plotting graph after each run
        size=10
        lab_el="Training loss"
        plt.plot(list(range(0,len(self.train_loss))), self.train_loss, 'r', label=lab_el)
        lab_vl="Validation loss"
        plt.plot(list(range(0,len(self.val_loss))), self.val_loss, 'b', label=lab_vl)
        topic="Loss vs Epochs"
        plt.title(topic, size=10)
        x_l="Epochs"
        plt.xlabel(x_l, size=10)
        plt.ylabel("Loss", size=10)
        plt.legend()
        plt.show()

    # opt.run((Xtrain,ytrain),(Xval,yval), momentum, beta, beta1, beta2, epsilon)

    def run(self,traindat,testdat,beta=val_1,beta1=val_1, beta2=val_2,epsilon=val_3):
        
        val_1=0.9
        val_2=0.999
        val_3=1e-10
        # choices
        _batch="batch"
        _sgd="sgd"
        _momentum="momentum"
        _nag="nag"
        _rmsprop="rmsprop"
        _adam="adam"
        _nadam="nadam"

        if self.optimizer==_batch:
            self.batch_gradient_descent(traindat,testdat)

        elif self.optimizer==_sgd:
            assert(self.batch_size==1), "stochastic gradient descent should have Batch size = 1 "
            self.batch_gradient_descent(traindat,testdat)

        elif self.optimizer==_momentum:
            self.momentum(traindat,testdat,beta)

        elif self.optimizer==_nag:
            self.NAG(traindat,testdat,beta)

        elif self.optimizer==_rmsprop:
            self.rmsprop(traindat,testdat,beta=0.9,epsilon=1e-10)

        elif self.optimizer==_adam:
            self.Adam(traindat,testdat,beta1=0.9, beta2=0.999,epsilon=1e-10)

        elif self.optimizer==_nadam:
            self.NAdam(traindat,testdat,beta1=val_1, beta2=val_2,epsilon=val_3)


        self.plot_loss() #Plot optimizer's call Plot



def one_hot(inarray): #hot encoding conversion
    outarray = np.zeros((inarray.size, inarray.max() + 1))
    outarray[np.arange(inarray.size), inarray] = 1
    return outarray

from keras.datasets import fashion_mnist



def dataset_scaling(X,y):
    div=255
    X_processed=np.reshape(X,(X.shape[0],784))/div
    X_processed=X_processed.T
    y_processed=one_hot(y).T
    return np.array(X_processed),y_processed

"""# TRAIN.PY"""

import argparse
from sklearn.model_selection import train_test_split

def main(args):
    
    X_size=784 #same for both datasets
    Y_size=10 #same for both datasets

    num_layers=args.num_layers
    const_hidden_layer_size=args.hidden_size
    const_hidden_layer_activation=args.activation.lower() #convert string to lower case for ease
    const_hidden_layer_initializations=args.weight_init.lower()#convert string to lower case for ease
    loss=args.loss
    optimizer=args.optimizer.lower()
    lamdba=args.weight_decay
    batch_size=args.batch_size
    epochs=args.epochs
    eta=args.learning_rate
    ES=args.earlystop
    log=args.logger
    dataset=args.dataset.lower()


    momentum=args.momentum
    beta=args.beta
    beta1=args.beta1
    beta2=args.beta2
    epsilon=args.epsilon

    #######################################################
    if dataset=='fashion_mnist':
        (X_train,y_train),(X_test,y_test)=fashion_mnist.load_data()
    else: #dataset=='mnist'
        (X_train,y_train),(X_test,y_test)=mnist.load_data()


    Xtest,ytest=dataset_scaling(X_test,y_test)

    opt=optimizers_wandb(X_size,Y_size,num_layers,const_hidden_layer_size,const_hidden_layer_activation,epsilon,
                    const_hidden_layer_initializations,loss,optimizer,lamdba,batch_size,epochs,eta,True,True)


    # if args.mode.lower()!='test':
    #     Xtrainfull,ytrainfull=dataset_scaling(X_train,y_train) ####Used for TESTING. DO NOT pass this to the model  FOR TRAINING!!!!
    #     opt.run((Xtrainfull,ytrainfull))
    # else:
    Xtrain,Xval,ytrain,yval=Xtrain,Xval,ytrain,yval=train_test_split(X_train,y_train,test_size=0.1) #test data size 10%
    Xtrain,ytrain=dataset_scaling(Xtrain,ytrain)
    Xval,yval=dataset_scaling(Xval,yval)
    opt.run((Xtrain,ytrain),(Xval,yval))


    # return opt.neural_Nnetwork_dl.predict(Xtest, args.probab)


    ###################################################################Argepass Block#######################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-wp' , '--wandb_project', help='Project name used to track experiments in Weights & Biases dashboard' , type=str, default='CS23M013_DL_A1')
    parser.add_argument('-we', '--wandb_entity' , help='Wandb Entity used to track experiments in the Weights & Biases dashboard.' , type=str, default='cs23m013')
    parser.add_argument('-d', '--dataset', help='choices: ["mnist", "fashion_mnist"]', choices = ["mnist", "fashion_mnist"],type=str, default='fashion_mnist')
    parser.add_argument('-e', '--epochs', help="Number of epochs to train neural network.", type=int, default=10)
    parser.add_argument('-b', '--batch_size', help="Batch size used to train neural network.", type=int, default=16)
    parser.add_argument('-l','--loss', help = 'choices: ["mean_squared_error", "cross_entropy"]' , choices = ["mean_squared_error", "cross_entropy"],type=str, default='cross_entropy')
    parser.add_argument('-o', '--optimizer', help = 'choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]', choices = ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],type=str, default = 'nadam')
    parser.add_argument('-lr', '--learning_rate', help = 'Learning rate used to optimize model parameters', type=float, default=0.001)
    parser.add_argument('-m', '--momentum', help='Momentum used by momentum and nag optimizers.',type=float, default=0.9)
    parser.add_argument('-beta', '--beta', help='Beta used by rmsprop optimizer',type=float, default=0.9)
    parser.add_argument('-beta1', '--beta1', help='Beta1 used by adam and nadam optimizers.',type=float, default=0.9)
    parser.add_argument('-beta2', '--beta2', help='Beta2 used by adam and nadam optimizers.',type=float, default=0.999)
    parser.add_argument('-eps', '--epsilon', help='Epsilon used by optimizers.',type=float, default=1e-8)
    parser.add_argument('-w_d', '--weight_decay', help='Weight decay used by optimizers.',type=float, default=0.5)
    parser.add_argument('-w_i', '--weight_init', help = 'choices: ["random", "Xavier"]', choices = ["random", "Xavier"],type=str, default='Xavier')
    parser.add_argument('-nhl', '--num_layers', help='Number of hidden layers used in feedforward neural network.',type=int, default=4)
    parser.add_argument('-sz', '--hidden_size', help ='Number of hidden neurons in a feedforward layer.', type=int, default=128)
    parser.add_argument("-ES","--earlystop",type=bool,default=True,help="Perform Early Stopping or not")
    parser.add_argument('-a', '--activation', help='choices: ["identity", "sigmoid", "tanh", "ReLU"]', choices = ["identity", "sigmoid", "tanh", "ReLU"],type=str, default='tanh')
    parser.add_argument("-lg","--logger",type=bool,default=True,help="Log to wandb or not")
    parser.add_argument("-md","--mode",type=str,default="test",help="Test mode, or train+val")
    parser.add_argument("-prb","--probab",type=bool,default=True,help="Test mode, or train+val")
    args = parser.parse_args()
    wandb.init(config=args, project = args.wandb_project, entity = args.wandb_entity, name = "cs23m013")
    main(args)
    wandb.finish()