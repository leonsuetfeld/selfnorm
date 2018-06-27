# cifar10 pre-processing

import numpy as np
import pickle
import scipy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# ##############################################################################
# ### FUNTIONS #################################################################
# ##############################################################################

def reshape_cifar(dataset):
    return dataset.reshape((dataset.shape[0],3,32,32)).transpose([0,2,3,1])

def show_images(dataset, n=5, random=False):
    if random:
        idcs = np.random.randint(dataset.shape[0], size=n**2).reshape(n,n)
    else:
        idcs = np.array(list(range(n**2))).reshape(n,n)
    plt.figure(figsize=(12,12))
    for i in range(n):
        for j in range(n):
            plt.subplot(n,n,i*n+j+1)
            img = dataset[idcs[i,j]]
            img -= np.min(img)
            img = img/np.max(img)
            a = plt.imshow(img)
            a.axes.get_xaxis().set_visible(False)
            a.axes.get_yaxis().set_visible(False)
    plt.show()

def GCN(dataset, s=1., lmda=0., epsilon=1.0e-8, goodfellow=False):
    """Per-image Global Contrast Normalization (except for scaling by factor s)
    """
    out_set = []
    print('\nGCN...')
    for i in range(dataset.shape[0]):
        X = np.squeeze(dataset[i,:,:,:])
        X_mean = np.mean(X)
        X = X - X_mean
        contrast = np.sqrt(lmda + np.mean(X**2))
        X = X / max(contrast, epsilon)
        out_set.append(X)
        print('%i / %i' %(i+1,dataset.shape[0]),end='\r')
    out_set = np.array(out_set)
    # Goodfellow mode
    if goodfellow:
        s_inv = np.zeros((32,32,3))
        for x in range(s_inv.shape[0]):
            for y in range(s_inv.shape[1]):
                for c in range(s_inv.shape[2]):
                    s_inv[x,y,c] = np.std(out_set[:,x,y,c])
        s = np.expand_dims(1./s_inv,0)
        out_set = out_set*s
    # return
    print('done.')
    return out_set

def ZCA(dataset, epsilon=0.0):
    """Zero-phased component analysis.
    Unfinished, may need revision.
    """
    # to do: epsilon? Keras has fixed 10e-7
    print('\nZCA step 1: compute PCA...')
    pca = PCA(n_components=3072, random_state=0, svd_solver='randomized')
    dataset_flattened = np.reshape(dataset,(np.shape(dataset)[0],-1))
    pca.fit(dataset_flattened)
    print('done.')
    out_set = []
    print('ZCA step 2: Whitening data...')
    for i in range(dataset.shape[0]):
        vec = dataset_flattened[i]
        dot = np.dot(vec - pca.mean_, pca.components_.T)
        whitened = np.dot(dot / (pca.singular_values_+epsilon), pca.components_) * np.sqrt(dataset.shape[0]) * 0.5 # added +epsilon
        out_set.append(whitened.reshape((32,32,3)))
        print('%i / %i' %(i+1,dataset.shape[0]),end='\r')
    print('done.')
    out_set = np.array(out_set)
    return out_set

def z_trans(dataset, bounded_std=True): # float 32 for testing purposes
    print('\nPerforming image-wise z-transformation...')
    dataset_new = []
    for i in range(dataset.shape[0]):
        img = dataset[i,:,:,:]
        img = img.astype(np.float32)
        img_mean = np.mean(img, dtype=np.float32)
        img_std = np.std(img, dtype=np.float32)
        if bounded_std:
            img_std = np.maximum(img_std,(1. / np.sqrt(3072.)), dtype=np.float32)
        img = np.divide((img-img_mean), img_std, dtype=np.float32)
        dataset_new.append(img)
    print('done.')
    return np.array(dataset_new)

def unify_dataset(dataset):
    ds_mean = np.mean(dataset)
    ds_std = np.std(dataset)
    return (dataset - ds_mean) / ds_std

def save_dataset(images, labels, path, filename):
    print('\nSaving file...')
    data_dict = { 'labels': labels, 'images': images }
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+filename, 'wb') as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('file saved: %s'%(path+filename))

def split_dataset(dataset_images, dataset_labels, splitpoint=50000):
    print('\nSplitting datasets into training and testing datasets...')
    training_images = dataset_images[:splitpoint,:,:,:]
    test_images = dataset_images[splitpoint:,:,:,:]
    training_labels = dataset_labels[:splitpoint]
    test_labels = dataset_labels[splitpoint:]
    print('done.')
    return training_images, training_labels, test_images, test_labels

def load_cifar10(path_train='./1_data_cifar10/train_batches/', path_test='./1_data_cifar10/test_batches/', max_size = 0):
    print('\nLoading and fusing cifar 10 training and test sets...')
    dataset_images = []
    dataset_labels = []
    for batch in range(5):
        with open(path_train+'data_batch_' + str(batch+1), 'rb') as file:
            data_dict = pickle.load(file, encoding='bytes')
            images = data_dict[b'data']
            dataset_images.extend(images)
            labels = data_dict[b'labels']
            dataset_labels.extend(labels)
    with open(path_test+'test_batch', 'rb') as file:
        test_dict = pickle.load(file, encoding='bytes')
        images = test_dict[b'data']
        dataset_images.extend(images)
        labels = test_dict[b'labels']
        dataset_labels.extend(labels)
    dataset_images = np.array(dataset_images)
    dataset_labels = np.array(dataset_labels)
    if max_size > 0:
        dataset_images = dataset_images[:max_size,:]
        dataset_labels = dataset_labels[:max_size]
    print('done.')
    return dataset_images, dataset_labels

def load_cifar100(path_train='./1_data_cifar100/train_batches/', path_test='./1_data_cifar100/test_batches/', max_size = 0):
    print('\nLoading and fusing cifar 100 training and test sets...')
    dataset_images = []
    dataset_labels = []
    with open(path_train+'train', 'rb') as file:
        data_dict = pickle.load(file, encoding='bytes')
        dataset_images.extend(data_dict[b'data'])
        dataset_labels.extend(data_dict[b'fine_labels'])
    with open(path_test+'test', 'rb') as file:
        test_dict = pickle.load(file, encoding='bytes')
        dataset_images.extend(test_dict[b'data'])
        dataset_labels.extend(test_dict[b'fine_labels'])
    dataset_images = np.array(dataset_images)
    dataset_labels = np.array(dataset_labels)
    if max_size > 0:
        dataset_images = dataset_images[:max_size,:]
        dataset_labels = dataset_labels[:max_size]
    print('done.')
    return dataset_images, dataset_labels

def array_to_list(a):
    if len(a.shape)==4:
        b = []
        for i in range(a.shape[0]):
            b.append(a[i,:,:,:])
    if len(a.shape)==1:
        b = a.tolist()
    return b

# ##############################################################################
# ### PROCESS DATA #############################################################
# ##############################################################################

if __name__ == '__main__':

    print('\n#############################################')
    print('### CIFAR 10 NO PRE-PROCESSING ##############')
    print('#############################################')

    # load and reshape images
    dataset_images, dataset_labels = load_cifar10()
    dataset_images = reshape_cifar(dataset_images)

    # separate training and test data, save as files
    train_imgs, train_lbls, test_imgs, test_lbls = split_dataset(dataset_images, dataset_labels, splitpoint=50000)
    train_imgs = array_to_list(train_imgs)
    train_lbls = array_to_list(train_lbls)
    test_imgs = array_to_list(test_imgs)
    test_lbls = array_to_list(test_lbls)
    save_dataset(train_imgs, train_lbls, './1_data_cifar10/train_set/', 'cifar10_trainset.pkl')
    save_dataset(test_imgs, test_lbls, './1_data_cifar10/test_set/', 'cifar10_testset.pkl')

    print('\n#############################################')
    print('### CIFAR 100 NO PRE-PROCESSING #############')
    print('#############################################')

    # load and reshape images
    dataset_images, dataset_labels = load_cifar100()
    dataset_images = reshape_cifar(dataset_images)

    # separate training and test data, save as files
    train_imgs, train_lbls, test_imgs, test_lbls = split_dataset(dataset_images, dataset_labels, splitpoint=50000)
    train_imgs = array_to_list(train_imgs)
    train_lbls = array_to_list(train_lbls)
    test_imgs = array_to_list(test_imgs)
    test_lbls = array_to_list(test_lbls)
    save_dataset(train_imgs, train_lbls, './1_data_cifar100/train_set/', 'cifar100_trainset.pkl')
    save_dataset(test_imgs, test_lbls, './1_data_cifar100/test_set/', 'cifar100_testset.pkl')

    print('\n#############################################')
    print('### CIFAR 10 Z-TRANS ########################')
    print('#############################################')

    # load and reshape images
    dataset_images, dataset_labels = load_cifar10()
    dataset_images = reshape_cifar(dataset_images)

    # pre-processing
    dataset_images = dataset_images.astype(np.float32) # set to float 32 before the calculations to test if this decreases performance by introducing noise
    dataset_images = z_trans(dataset_images)

    # separate training and test data, save as files
    train_imgs, train_lbls, test_imgs, test_lbls = split_dataset(dataset_images, dataset_labels, splitpoint=50000)
    train_imgs = array_to_list(train_imgs)
    train_lbls = array_to_list(train_lbls)
    test_imgs = array_to_list(test_imgs)
    test_lbls = array_to_list(test_lbls)
    save_dataset(train_imgs, train_lbls, './1_data_cifar10/train_set_ztrans/', 'cifar10_trainset.pkl')
    save_dataset(test_imgs, test_lbls, './1_data_cifar10/test_set_ztrans/', 'cifar10_testset.pkl')

    print('\n#############################################')
    print('### CIFAR 100 Z-TRANS #######################')
    print('#############################################')

    # load and reshape images
    dataset_images, dataset_labels = load_cifar100()
    dataset_images = reshape_cifar(dataset_images)

    # pre-processing
    dataset_images = z_trans(dataset_images)

    # separate training and test data, save as files
    train_imgs, train_lbls, test_imgs, test_lbls = split_dataset(dataset_images, dataset_labels, splitpoint=50000)
    train_imgs = array_to_list(train_imgs)
    train_lbls = array_to_list(train_lbls)
    test_imgs = array_to_list(test_imgs)
    test_lbls = array_to_list(test_lbls)
    save_dataset(train_imgs, train_lbls, './1_data_cifar100/train_set_ztrans/', 'cifar100_trainset.pkl')
    save_dataset(test_imgs, test_lbls, './1_data_cifar100/test_set_ztrans/', 'cifar100_testset.pkl')

    print('\n#############################################')
    print('### CIFAR 10 GCN & ZCA PRE-PROCESSING #######')
    print('#############################################')

    # load and reshape images
    dataset_images, dataset_labels = load_cifar10()
    dataset_images = reshape_cifar(dataset_images)

    # perform GCN and ZCA
    dataset_images = GCN(dataset_images, goodfellow=True)
    dataset_images = ZCA(dataset_images)
    dataset_images = unify_dataset(dataset_images)
    print('dataset mean & std:', np.mean(dataset_images), np.std(dataset_images))

    # separate training and test data, save as files
    train_imgs, train_lbls, test_imgs, test_lbls = split_dataset(dataset_images, dataset_labels, splitpoint=50000)
    train_imgs = array_to_list(train_imgs)
    train_lbls = array_to_list(train_lbls)
    test_imgs = array_to_list(test_imgs)
    test_lbls = array_to_list(test_lbls)
    save_dataset(train_imgs, train_lbls, './1_data_cifar10/train_set_gcn_zca/', 'cifar10_trainset.pkl')
    save_dataset(test_imgs, test_lbls, './1_data_cifar10/test_set_gcn_zca/', 'cifar10_testset.pkl')

    print('\n#############################################')
    print('### CIFAR 100 GCN & ZCA PRE-PROCESSING ######')
    print('#############################################')

    # load and reshape images
    dataset_images, dataset_labels = load_cifar100()
    dataset_images = reshape_cifar(dataset_images)

    # perform GCN and ZCA
    dataset_images = GCN(dataset_images, goodfellow=True)
    dataset_images = ZCA(dataset_images)
    dataset_images = unify_dataset(dataset_images)
    print('dataset mean & std:', np.mean(dataset_images), np.std(dataset_images))

    # separate training and test data, save as files
    train_imgs, train_lbls, test_imgs, test_lbls = split_dataset(dataset_images, dataset_labels, splitpoint=50000)
    train_imgs = array_to_list(train_imgs)
    train_lbls = array_to_list(train_lbls)
    test_imgs = array_to_list(test_imgs)
    test_lbls = array_to_list(test_lbls)
    save_dataset(train_imgs, train_lbls, './1_data_cifar100/train_set_gcn_zca/', 'cifar100_trainset.pkl')
    save_dataset(test_imgs, test_lbls, './1_data_cifar100/test_set_gcn_zca/', 'cifar100_testset.pkl')

    print('\n#############################################')
    print('### ALL FINISHED ############################')
    print('#############################################\n')

    show_images(dataset_images)
