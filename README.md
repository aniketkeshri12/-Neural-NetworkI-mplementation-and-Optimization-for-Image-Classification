#  Neural Network Implementation and Optimization for ImageC lassification

# Deep Learning Assignment 1: Neural Network development

##### ANIKET KESHRI CS23M013

This is assignment number 1 in the course, Fundamentals of Deep Learning CS6910 by Prof. Mitesh Khapra.
We implement a feed forward neural network and use verious flavors of the gradient descent algorithm such as momentum, nesterov, RMSprop, Adam, NAdam  and compare  them. The datasets chosen are fashion-mnist and mnist, which have images of size 28x28 and 10 classes.

We run upto 390 different configurations and track them all using wandb, we then find correlations with the best features and tune further searches to attempt to reach as high an accuracy as possible:-

Report can be accessed here:- https://wandb.ai/cs23m013/CS23M013_DL_A1/reports/CS6910-Assignment-1--Vmlldzo3MTc1OTU1


### Libraries used:
- copy was used to obtain a deep copy of the class Model
- tqdm was used to track time left in a particular run
- wandb was used to log all the runs with their metrics
- matplotlib and seaborn was used to plot graphs such as confusion matrix, ROC cuvres


### Dataset
Fashion MNIST data set has been used here in this assignment instead of the traditional MNIST hand written digits dataset. Train - 60000 Test - 20000 Validation - 6000

For the hyper parameter optimisation stage, 10% of the randomly shuffled training data set (around 6000 images and corresponding labels) are kept aside for validation for each hyperparameter configuration while the model is trained on the remaining 54000 images from the randomly shuffled training data set.

Once the best configuration is identified with the help of wandb wither using Random search or Bayesian optimisation, the full training dataset is used to train the best model configuration and the test accuracy is calculated. The resulting confusion matrix is plotted therafter.


### Neural network
The Neural network multiple method with implementations for Feedforward network and a Backpropagation method Alog with training which can batch train with customizable batch sizes. The neural net has features with different optimisation functions such as SGD , Nag , adam, nadam etc. also multiple activation functions such as sigmoid tanh and ReLU. This network mainly is made for Mnist datasets and for now work with Mnist and Fashion_mnist datasets.

#### classes involves

1. class all_classes : classify the dataset into 10 classes
2. class layer : will initiate the initial values of parameters 
3. class Neural_Network_DL : contains implementations of forward and backward propogation
4. class optimizers_wandb: will contain various optimizers like nadam, adam,sgd e.t.c

### Optimizers
These are the optimizers implemented

batch_gradient_descent: This function implements Mini-Batch Gradient Descent. It updates the weights and biases of the neural network by taking the average of the gradients of a small batch of data. The size of the batch can be changed by changing the value of batchsize parameter.

momentum: This function implements Momentum-Based Gradient Descent. It adds a fraction of the previous gradient to the current gradient. This helps the algorithm to avoid oscillations and converge faster.

rmsprop: This function implements Root Mean Squared Propagation. It scales the learning rate by diving it by the square root of a running average of the squares of the gradients

Adam: This function implements Adaptive Moment Estimation. It combines the advantages of both momentum-based and rmsprop-based optimization algorithms. It maintains an exponentially decaying average of past gradients and squared gradients, and calculates the update step based on these moving averages.

NAG: This function implements Nesterov Accelarated Gradient descent. It is a modification of the Momentum-Based Gradient Descent algorithm. It first calculates an approximate position of the next step, then calculates the gradient of the cost function at that position and finally updates the weights and biases of the neural network based on that gradient.

NAdam: This function implemets the NAdam optimizer. Like Adam, NAdam keeps track of the first and second moments of the gradients, but it also incorporates the Nesterov 'look ahead' idea to update the weights. NAdam also includes bias correction in its update rule, which helps to correct the bias that could be introduced in the initial iterations


## train.py

The train.py script is used to train the neural network using the optimizer class with various options that can be specified using command-line arguments. Here is a description of the available options. The deafult values are set according to what worked best in the wandb sweeps.

    parser.add_argument('-wp' , '--wandb_project', help='Project name used to track experiments in Weights & Biases dashboard' , type=str, default='CS23M013_DL_A1')
    parser.add_argument('-we', '--wandb_entity' , help='Wandb Entity used to track experiments in the Weights & Biases dashboard.' , type=str, default='cs23m013')
    parser.add_argument('-d', '--dataset', help='choices: ["mnist", "fashion_mnist"]', choices = ["mnist", "fashion_mnist"],type=str, default='fashion_mnist')
    parser.add_argument('-e', '--epochs', help="Number of epochs to train neural network.", type=int, default=10)
    parser.add_argument('-b', '--batch_size', help="Batch size used to train neural network.", type=int, default=64)
    parser.add_argument('-l','--loss', help = 'choices: ["mean_squared_error", "cross_entropy"]' , choices = ["mean_squared_error", "cross_entropy"],type=str, default='cross_entropy')
    parser.add_argument('-o', '--optimizer', help = 'choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]', choices = ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],type=str, default = 'adam')
    parser.add_argument('-lr', '--learning_rate', help = 'Learning rate used to optimize model parameters', type=float, default=0.0005)
    parser.add_argument('-m', '--momentum', help='Momentum used by momentum and nag optimizers.',type=float, default=0.9)
    parser.add_argument('-beta', '--beta', help='Beta used by rmsprop optimizer',type=float, default=0.9)
    parser.add_argument('-beta1', '--beta1', help='Beta1 used by adam and nadam optimizers.',type=float, default=0.9)
    parser.add_argument('-beta2', '--beta2', help='Beta2 used by adam and nadam optimizers.',type=float, default=0.999)
    parser.add_argument('-eps', '--epsilon', help='Epsilon used by optimizers.',type=float, default=1e-8)
    parser.add_argument('-w_d', '--weight_decay', help='Weight decay used by optimizers.',type=float, default=0.0005)
    parser.add_argument('-w_i', '--weight_init', help = 'choices: ["random", "Xavier"]', choices = ["random", "Xavier"],type=str, default='random')
    parser.add_argument('-nhl', '--num_layers', help='Number of hidden layers used in feedforward neural network.',type=int, default=4)
    parser.add_argument('-sz', '--hidden_size', help ='Number of hidden neurons in a feedforward layer.', type=int, default=128)
    parser.add_argument("-ES","--earlystop",type=bool,default=True,help="Perform Early Stopping or not")
    parser.add_argument('-a', '--activation', help='choices: ["identity", "sigmoid", "tanh", "ReLU"]', choices = ["identity", "sigmoid", "tanh", "ReLU"],type=str, default='ReLU')
    parser.add_argument("-lg","--logger",type=bool,default=True,help="Log to wandb or not")
    parser.add_argument("-md","--mode",type=str,default="test",help="Test mode, or train+val")
    parser.add_argument("-prb","--probab",type=bool,default=True,help="Test mode, or train+val")
  
