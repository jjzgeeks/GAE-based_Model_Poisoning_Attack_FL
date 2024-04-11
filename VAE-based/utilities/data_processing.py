import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import tensorflow as tf



def create_kmeans_clusters(X, Y, n_clusters, random_state = 0):
  clusters = KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(X)
  result = []
  for i in range(n_clusters):
    result.append(X[clusters == i])
    result.append(Y[clusters == i])
  return tuple(result)



def load_mnist_return_required_digits(n1, n2):
  # Loading the mnist dataset and concatenating train - test sets
  #(x_train, y_train), (x_test, y_test) =  tf.keras.datasets.cifar10.load_data()
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  x_total = np.concatenate((x_train, x_test), axis=0)
  y_total = np.concatenate((y_train, y_test), axis=0)

  # Normalizing and reshaping the data
  x_total = x_total/255
  x_total = x_total.reshape(x_total.shape[0],784) # x_total.shape[0] = 70000, 784 is the number features of image
  #x_total = x_total.reshape(x_total.shape[0], 32, 32, 3)
  #x_total = x_total.reshape (x_total, (x_total.shape[0], -1))
  x_n1 = x_total[y_total == n1]
  y_n1 = y_total[y_total == n1]
  x_n2 = x_total[y_total == n2]
  y_n2 = y_total[y_total == n2]
  return [(x_n1, y_n1), (x_n2, y_n2)]


"""This is a dataset of 60,000 28x28 grayscale images of 10 fashion categories,
 along with a test set of 10,000 images. This dataset can be used as a drop-in 
 replacement for MNIST."""
def load_fashion_mnist_return_required_digits(n1, n2):
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
  x_total = np.concatenate((x_train, x_test), axis=0)
  y_total = np.concatenate((y_train, y_test), axis=0)

  # Normalizing and reshaping the data
  x_total = x_total/255
  x_total = x_total.reshape(x_total.shape[0],784) # x_total.shape[0] = 70000, 784 is the number features of image
  x_n1 = x_total[y_total == n1]
  y_n1 = y_total[y_total == n1]
  x_n2 = x_total[y_total == n2]
  y_n2 = y_total[y_total == n2]
  return [(x_n1, y_n1), (x_n2, y_n2)]



def rgb2gray(rgb):
    """Convert from color image (RGB) to grayscale.
       Source: opencv.org
       grayscale = 0.299*red + 0.587*green + 0.114*blue
    Argument:
        rgb (tensor): rgb image
    Return:
        (tensor): grayscale image
    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

"""The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes,
 with 6000 images per class. There are 50000 training images and 10000 test images."""
def load_cifar10_return_required_digits(m, n):
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  # convert color train and test images to gray
  # x_train shape is: (50000, 32, 32, 3); x_train_gray shape is (50000, 32, 32) 5000 pictures of 32*32
  x_train_gray = rgb2gray(x_train)
  x_test_gray = rgb2gray(x_test)
  x_total = np.concatenate((x_train_gray, x_test_gray), axis=0)
  y_total = np.concatenate((y_train, y_test), axis=0)
  y_total = y_total.flatten()  # Flatten a 2d numpy array into 1d array. y_total shape is  (60000,)

  # Normalizing and reshaping the data
  x_total = x_total.astype('float32') / 255
  x_total = x_total.reshape(x_total.shape[0], 1024)  # x_total shape is (60000, 1024)

  x_n1 = x_total[y_total == m]
  y_n1 = y_total[y_total == m]
  x_n2 = x_total[y_total == n]
  y_n2 = y_total[y_total == n]
  # print(y_total == 1) # [False False False ... False  True False]
  return [(x_n1, y_n1), (x_n2, y_n2)]




def get_clients(class1, class2, n_clients):
  clients_X = []
  clients_y = []

  clientsXtest = []
  clientsYtest = []

  clusters_1 = KMeans(n_clusters=n_clients, random_state=0).fit_predict(class1)
  clusters_2 = KMeans(n_clusters=n_clients, random_state=0).fit_predict(class2)


  for i in range(n_clients):
    X_train0, X_test0, y_train0, y_test0 = train_test_split(class1[clusters_1 == i],np.zeros((class1[clusters_1 == i].shape[0],)),test_size=0.2)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(class2[clusters_2 == i],np.ones((class2[clusters_2 == i].shape[0],)),test_size=0.2)

    clients_X.append([X_train0, X_train1])
    clients_y.append([y_train0, y_train1])
    clientsXtest.extend([X_test0,X_test1])
    clientsYtest.extend([y_test0,y_test1])

  X_test = np.concatenate(clientsXtest,axis=0)
  y_test = np.concatenate(clientsYtest,axis=0)

  return clients_X,clients_y,X_test,y_test





def get_total_from_clients(clients_X, clients_y):
  x_train0 = [i[0] for i in clients_X]
  x_train0 = np.concatenate(x_train0, axis=0)
  x_train1 = [i[1] for i in clients_X]
  x_train1 = np.concatenate(x_train1, axis=0)
  y_train0 = [i[0] for i in clients_y]
  y_train0 = np.concatenate(y_train0, axis=0)
  y_train1 = [i[1] for i in clients_y]
  y_train1 = np.concatenate(y_train1, axis=0)
  return ([x_train0,x_train1],[y_train0,y_train1])