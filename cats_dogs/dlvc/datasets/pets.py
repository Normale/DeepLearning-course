from ..dataset import Sample, Subset, ClassificationDataset
import pickle
import numpy as np
import os

class PetsDataset(ClassificationDataset):
    '''
    Dataset of cat and dog images from CIFAR-10 (class 0: cat, class 1: dog).
    '''

    def __init__(self, fdir: str, subset: Subset):
        '''
        Loads a subset of the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all cat and dog images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all cat and dog images from "data_batch_5".
          - The test set contains all cat and dog images from "test_batch".

        Images are loaded in the order the appear in the data files
        and returned as uint8 numpy arrays with shape (32, 32, 3), in BGR channel order.
        '''

        # TODO implement
        # See the CIFAR-10 website on how to load the data files
        
        def unpickle(file):
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict
        
        
        f_names = []
        if subset == Subset.TRAINING:
            for i in range(1, 5):
                f_names.append(os.path.join(fdir, 'data_batch_{}'.format(i)))
        elif subset == Subset.VALIDATION:
            f_names.append(os.path.join(fdir, 'data_batch_5'))
        elif subset == Subset.TEST:
            f_names.append(os.path.join(fdir, 'test_batch'))

        for file in f_names:
            if not os.path.exists(file):
                raise ValueError("Either fdir is not a directory or there is no file in it.")
        
        cifar_dicts=[]
        for file in f_names:
            cifar_dicts.append(unpickle(file))
        
        images_raw = []
        labels_raw = []
        for diz in cifar_dicts:
            images_raw.extend(diz[b'data'])
            labels_raw.extend(diz[b'labels'])
            
        images = []
        labels = []
        
        def _convert_imgs(raw):
            img = raw.reshape(3, 32, 32)
            img = np.einsum('abc->bca', img)
            return img[..., ::-1]
        
        for i in range(len(labels_raw)):
            if labels_raw[i] == 3:  # cats 
                labels.append(0)
                images.append(np.asarray(_convert_imgs(images_raw[i])))
            elif labels_raw[i] == 5:  # dogs 
                labels.append(1)
                images.append(np.asarray(_convert_imgs(images_raw[i])))

        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''

        return len(self.images)


    def __getitem__(self, idx: int) -> Sample:
        '''
        Returns the idx-th sample in the dataset.
        Raises IndexError if the index is out of bounds. Negative indices are not supported.
        '''

        if idx >= self.__len__() or idx < 0:
            raise IndexError("Index out of bounds: valid input range:{} while provided index:{}".format([0, self.__len__()-1], idx))
        else:
            return Sample(idx, self.images[idx], self.labels[idx])

        pass

    def num_classes(self) -> int:
        '''
        Returns the number of classes.
        '''
        return len(np.unique(self.labels))
