import numpy as np
from sklearn.preprocessing import OneHotEncoder

def unpickle(file):
    """
    Each of the files from the CIFAR-10 dataset is a Python "pickled" object produced with cPickle
    This routine opens such file and returns a dictionary
    """
    
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
	

def img_converter(raw):
    """
    Converts images from the CIFAR-10 format and returns a 4-dim array with shape:
    [image_number, height, width, channel] where the pixels are floats between 0.0 and 1.0
    """

    # Convert raw images to floats
    raw_conv = np.array(raw, dtype=float) / 255.0
    # Image dimensions 
    n_channels = 3
    img_height = 32
    img_width = 32
	# Reshape
    img = raw_conv.reshape([-1, n_channels, img_height, img_width])
    # Reorder the indices of the array so that the number of channels is last
    img = img.transpose([0, 2, 3, 1])

    return img
	
def load_classes():
    """
    Loads class labels and returns a list with class names
    """

    # Load from pickled file
    raw = unpickle(file='cifar-10-batches-py/batches.meta')[b'label_names']
    # Convert from binary strings
    names = [x.decode('utf-8') for x in raw]

    return names

def load_data(file):
    """
    Loads dataset and returns converted images and respective class
    """

    # Get data
    data = unpickle(file)
    raw = data[b'data']
    target = np.array(data[b'labels'])
    img = img_converter(raw)

    return img, target

def load_training_data():
    """
    Loads training data from all data files and returns a tuple of images and respective classes
    """
    
    n_channels = 3
    img_height = 32
    img_width = 32
    #img = np.zeros(shape=[1, img_height, img_width, n_channels], dtype=float)
    img = np.array([])
    #target = np.zeros(shape=[50000], dtype=int)
    target = np.array([])
    n_files = 5
    begin = 0

    for i in range(n_files):
        img_batch, target_batch = load_data(file = 'cifar-10-batches-py/data_batch_' + str(i+1))
        n_img = len(img_batch)
        end = begin + n_img
        #img[begin:end, :] = img_batch
        img = np.append(img, img_batch)
        img = img.reshape([n_img*(i+1), img_height, img_width, n_channels])
        #target[begin:end] = target_batch
        target = np.append(target, target_batch)
        begin = end

        target = np.expand_dims(target, axis=-1)
        enc = OneHotEncoder(categories='auto')
        fit = enc.fit(target)
        ohe = fit.transform(target).toarray()
        print(img_batch.shape)
        print(target_batch.shape)
        print(img.shape)
        print(target.shape)
    return img, target, ohe

def load_test_data():
    """
    Load test data from all data files and returns a tuple of images and respective classes
    """

    img, target = load_data(file='cifar-10-batches-py/test_batch')

    target = np.expand_dims(target, axis=-1)
    enc = OneHotEncoder(categories='auto')
    fit = enc.fit(target)
    ohe = fit.transform(target).toarray()
    
    return img, target, ohe

