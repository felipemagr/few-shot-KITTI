from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import numpy as np
from PIL import Image as pil_image
import pickle
from . import parser


class Kitti(data.Dataset):
    def __init__(self, root, dataset='kitti'):
        self.root = root
        self.dataset = dataset
        if not self._check_exists_():
            self._init_folders_()
            if self.check_decompress():
                self._decompress_()
            self._preprocess_()

    def _init_folders_(self):
        decompress = False
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        if not os.path.exists(os.path.join(self.root, 'kitti')):
            os.makedirs(os.path.join(self.root, 'kitti'))
            decompress = True
        if not os.path.exists(os.path.join(self.root, 'kitti', 'training')):
            os.makedirs(os.path.join(self.root, 'kitti', 'training'))
            decompress = True
        if not os.path.exists(os.path.join(self.root, 'kitti', 'testing')):
            os.makedirs(os.path.join(self.root, 'kitti', 'testing'))
            decompress = True
        if not os.path.exists(os.path.join(self.root, 'compacted_datasets')):
            os.makedirs(os.path.join(self.root, 'compacted_datasets'))
            decompress = True
        return decompress

    def check_decompress(self):
        return os.listdir('%s/kitti/testing' % self.root) == []

    def _decompress_(self):
        print("\nDecompressing Images...")
        compressed_file = '%s/compressed/kitti/data_tracking_image_2.zip' % self.root
        compressed_file_labels = '%s/compressed/kitti/data_tracking_label_2.zip' % self.root
        if (os.path.isfile(compressed_file) and os.path.isfile(compressed_file_labels)):
            os.system('unzip %s -d %s/kitti/' % (compressed_file, self.root))
            os.system('unzip %s -d %s/kitti/' % (compressed_file_labels, self.root))
        else:
            raise Exception('Missing compressed files')
        print("Decompressed")

    def _check_exists_(self):
        if not os.path.exists(os.path.join(self.root, 'compacted_datasets', 'kitti_train.pickle')) or not \
                os.path.exists(os.path.join(self.root, 'compacted_datasets', 'kitti_test.pickle')):
            return False
        else:
            return True

    def get_image_paths(self):
        class_names, images_path = [], []
        folders = "0000"
        while int(folders) < 20:    # TRAINING
            file_name = self.root + '/kitti/training/label_02/' + folders + '.txt'
            with open(file_name, 'r') as f:
                f.readline()
                for line in f:
                    columns = line.split(',')
                    class_ = columns[2::17]
                    for class_name_element in class_:
                        class_names.append(class_name_element)
            path = os.listdir(self.root + '/kitti/training/image_02/'+ folders)
            for path_image in path:
                images_path.append(path_image)
            if (int(folders) < 9):
                folders = '000'+str(int(folders)+1)
            else:
                folders = '00'+str(int(folders)+1)
        return class_names, images_path

    def _preprocess_(self):
        print('\nPreprocessing KITTI images...')
        (class_names_train, images_path_train) = self.get_image_paths()
        (class_names_test, images_path_test) = parser.get_image_paths(os.path.join(self.root, 'kitti', 'testing'))

        keys_train = list(set(class_names_train))
        keys_test = list(set(class_names_test))

        label_encoder = {}
        label_decoder = {}
        for i in range(len(keys_train)):
            label_encoder[keys_train[i]] = i
            label_decoder[i] = keys_train[i]
        for i in range(len(keys_train), len(keys_train)+len(keys_test)):
            label_encoder[keys_test[i-len(keys_train)]] = i
            label_decoder[i] = keys_test[i-len(keys_train)]

        counter = 0
        train_set = {}
        for class_, path in zip(class_names_train, images_path_train):
            img = pil_image.open(path)
            img = img.convert('RGB')
            img = img.resize((84,84), pil_image.ANTIALIAS)
            img = np.array(img, dtype='float32')
            if label_encoder[class_] not in train_set:
                train_set[label_encoder[class_]] = []
            train_set[label_encoder[class_]].append(img)
            counter += 1
            if counter % 1000 == 0:
                print("Counter training "+str(counter) + " from " + str(len(images_path_train) + len(class_names_test)))

        test_set = {}
        for class_, path in zip(class_names_test, images_path_test):
            img = pil_image.open(path)
            img = img.convert('RGB')
            img = img.resize((84,84), pil_image.ANTIALIAS)
            img = np.array(img, dtype='float32')

            if label_encoder[class_] not in test_set:
                test_set[label_encoder[class_]] = []
            test_set[label_encoder[class_]].append(img)
            counter += 1
            if counter % 1000 == 0:
                print("Counter testing " + str(counter) + " from "+str(len(images_path_train) + len(class_names_test)))

        with open(os.path.join(self.root, 'compacted_datasets', 'kitti_train.pickle'), 'wb') as handle:
            pickle.dump(train_set, handle, protocol=2)
        with open(os.path.join(self.root, 'compacted_datasets', 'kitti_test.pickle'), 'wb') as handle:
            pickle.dump(test_set, handle, protocol=2)

        label_encoder = {}
        keys = list(train_set.keys()) + list(test_set.keys())
        for id_key, key in enumerate(keys):
            label_encoder[key] = id_key
        with open(os.path.join(self.root, 'compacted_datasets', 'kitti_label_encoder.pickle'), 'wb') as handle:
            pickle.dump(label_encoder, handle, protocol=2)

        print('Images preprocessed')

    def load_dataset(self, partition, size=(84,84)):
        print("Loading dataset")
        if partition == 'train':
            with open(os.path.join(self.root, 'compacted_datasets', 'kitti_%s.pickle' % 'train'),
                      'rb') as handle:
                data = pickle.load(handle)
            with open(os.path.join(self.root, 'compacted_datasets', 'kitti_%s.pickle' % 'test'),
                      'rb') as handle:
                data_val = pickle.load(handle)
            data.update(data_val)
            del data_val
        else:
            with open(os.path.join(self.root, 'compacted_datasets', 'kitti_%s.pickle' % partition),
                      'rb') as handle:
                data = pickle.load(handle)

        with open(os.path.join(self.root, 'compacted_datasets', 'kitti_label_encoder.pickle'),
                  'rb') as handle:
            label_encoder = pickle.load(handle)

        # Resize images and normalize
        for class_ in data:
            for i in range(len(data[class_])):
                image2resize = pil_image.fromarray(np.uint8(data[class_][i]))
                image_resized = image2resize.resize((size[1], size[0]))
                image_resized = np.array(image_resized, dtype='float32')

                # Normalize
                image_resized = np.transpose(image_resized, (2, 0, 1))
                image_resized[0, :, :] -= 120.45  # R
                image_resized[1, :, :] -= 115.74  # G
                image_resized[2, :, :] -= 104.65  # B
                image_resized /= 127.5

                data[class_][i] = image_resized
        print("Num classes " + str(len(data)))
        num_images = 0
        for class_ in data:
            num_images += len(data[class_])
        print("Num images " + str(num_images))
        return data, label_encoder