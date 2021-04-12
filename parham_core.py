
import os
import gdown
import tarfile
import imageio
from pathlib import Path
from shutil import copyfile

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_agg import FigureCanvasAgg

from torchviz import make_dot
from torch.utils.data import DataLoader, dataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.models as models
import torchvision.datasets as ds
import torchvision.transforms as trans

from poutyne import Callback, Dict, CSVLogger, Experiment

os.environ["CUDA_VISIBLE_DEVICES"]="0"

__result_dir__ = 'results'

def save_model(model, file_path):
    """save_model saves the trained model in the specified path

    Args:
        model (nn.Module): the model to be saved.
        file_path (str): file path.
    """

    torch.save(model, file_path)

def load_model(file_path):
    """load_model loads the saved model from specified path.

    Args:
        file_path (str): file path to the trained model.

    Returns:
        nn.Module: the trained model.
    """
    return torch.load(file_path)

def get_torch_device(): return torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu')

is_cuda_available = lambda: torch.cuda.is_available()

def get_cub200(dir_path: str):
    # Dowload the dataset
    url = 'https://drive.google.com/uc?id=1GDr1OkoXdhaXWGA8S3MAq3a522Tak-nx'
    dataset_file = 'images.tgz'
    
    print('Create required directories ...')
    # Make the original dataset directories
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    ds_zipfile = os.path.join(dir_path, dataset_file)

    if not os.path.exists(ds_zipfile):
        print('Download the dataset zip file ...')
        gdown.download(url, output=ds_zipfile, quiet=False)
        # Extract the tgz file into the dataset directory
        print('Extract the zip file ...')
        tar = tarfile.open(ds_zipfile, "r:gz")
        tar.extractall(path=dir_path)
        tar.close()

    return os.path.join(dir_path, 'images')


def prepare_subsets(orig_ds_dir : str, target_dir : str):
    # Make the dataset directory
    training_ds_dir = os.path.join(target_dir, 'training')
    testing_ds_dir = os.path.join(target_dir, 'testing')
    
    if os.path.isdir(orig_ds_dir):
        print('Make required folder for the dataset ...')
        Path(training_ds_dir).mkdir(parents=True, exist_ok=True)
        Path(testing_ds_dir).mkdir(parents=True, exist_ok=True)
        # Prepare the training and testing subsets
        print('Preparing the training and testing subsets ...')
        separate_train_test(orig_ds_dir, training_ds_dir, testing_ds_dir)

    return training_ds_dir, testing_ds_dir


def make_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def separate_train_test(dataset_path, train_path, test_path):
    class_index = 1
    for classname in sorted(os.listdir(dataset_path)):
        if classname.startswith('.'):
            continue
        make_dir(os.path.join(train_path, classname))
        make_dir(os.path.join(test_path, classname))
        i = 0
        for file in sorted(os.listdir(os.path.join(dataset_path, classname))):
            if file.startswith('.'):
                continue
            file_path = os.path.join(dataset_path, classname, file)
            if i < 15:
                copyfile(file_path, os.path.join(test_path, classname, file))
            else:
                copyfile(file_path, os.path.join(train_path, classname, file))
            i += 1

        class_index += 1

ds_filename_valid = lambda filename : not os.path.basename(filename).startswith('.')

def create_model(pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    model = model.cuda() if is_cuda_available() else model
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 200)
    model.fc = model.fc.cuda() if is_cuda_available() else model.fc

    return model

def calculate_normalization_params(dataset_dir : str):
    transforms = trans.Compose([
        trans.Resize((224,224)),
        trans.ToTensor()
    ])

    dataset = ds.ImageFolder(root=dataset_dir, transform=transforms, is_valid_file=ds_filename_valid)
    dl = DataLoader(dataset, batch_size=64, num_workers=2)

    count = 0
    first_moment = torch.empty(3)
    second_moment = torch.empty(3)

    for images, _ in dl:
        b, _, h, w = images.shape
        nump = b * h * w
        sumv = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        first_moment = (count * first_moment + sumv) / (count + nump)
        second_moment = (count * second_moment + sum_of_square) / (count + nump)
        count += nump

    mean_ = first_moment
    std_ = torch.sqrt(second_moment - first_moment ** 2)

    return mean_, std_

def get_datasets(ds_path, transforms, batch_size=32, num_workers=2, shuffle=True):
    dataset = ds.ImageFolder(root=ds_path, transform=transforms, 
        is_valid_file=ds_filename_valid)
    data_loader = DataLoader(dataset, batch_size=batch_size, 
        num_workers=num_workers, shuffle=shuffle)

    return dataset, data_loader

def run_experiment(
    name,
    model,
    transforms,
    training_dir,
    testing_dir,
    learning_rate=0.01,
    momentum=0.9,
    num_epochs=10,
    batch_size=32
):
    _, training_loader = get_datasets(training_dir, transforms, batch_size=batch_size)
    _, testing_loader = get_datasets(testing_dir, transforms)

    # Optimizer Initialization
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    # Loss Function Initialization
    loss_function = nn.CrossEntropyLoss()

    result_dir = os.path.join(__result_dir__,name)
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    res_photo = os.path.join(result_dir, 'result.png')
    if os.path.isfile(res_photo):
        img = mpimg.imread(res_photo)
        plt.imshow(img)
        return

    callbacks = [
        PlottingCallback(result_dir=result_dir),
        # Save the losses and accuracies for each epoch in a TSV.
        CSVLogger(os.path.join(result_dir, 'log_callback.tsv'), separator='\t'),
    ]

    # Create & Initialize Experiment
    exp_obj = Experiment(
        result_dir, model,
        device=get_torch_device(),
        optimizer=optimizer,
        loss_function=loss_function,
        batch_metrics=['accuracy'],
        epoch_metrics=['f1'],
        task='classif'
    )

    # Use Poutyne experiment to train the model.
    exp_obj.train(train_generator=training_loader, valid_generator=testing_loader,
                epochs=num_epochs, callbacks=callbacks)

    print(exp_obj.test(testing_loader))

    # Save the trained model
    save_model(model, os.path.join(result_dir, name + '.phm'))

def freeze_conv_params(model):
    for name, param in model.named_parameters():
        if name.startswith('conv'):
            param.requires_grad = False
    return model

def freeze_layer1_params(model):
    for name, param in model.named_parameters():
        if name.startswith('layer1'):
            param.requires_grad = False
    return model
    # for param in model.layer1.parameters():
    #     param.requires_grad = False

    # return model

class PlottingCallback(Callback):
    """ PhmPlottingCallback is the class containing the codes for visualizing the model's results. 
        The class also extends Callback class makes it able to attach to Poutyne experiment and register for the raised events.
    """

    def __init__(self, result_dir='./'):
        super().__init__()
        self.report = list()
        self.fig_list = list()
        self.fig, (self.loss_ax, self.acc_ax) = plt.subplots(2, 1)
        self.fig.tight_layout(pad=1.0)
        self.fig.suptitle('Statistics of training model')
        self.result_dir = result_dir
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        plt.ion()
        # plt.show()

    def on_train_begin(self, logs: Dict):
        """ on_train_begin is a method invoked in the begining of each train. """
        pass

    def _plot(self, axis, epoch, train_data, valid_data, label, maxv=1.0, minv=0.0, stepv=0.05):
        axis.clear()
        plt.yticks(fontsize=11)
        axis.plot(epoch, train_data, '.-', label='Train')
        axis.plot(epoch, valid_data, '.-', label='Validation')
        axis.set_xlabel('epoch')
        axis.set_ylabel(label)
        # axis.set_yticks(np.arange(minv, maxv, stepv), minor = True)
        axis.set_yticks(np.arange(min(train_data + valid_data),
                                  max(train_data + valid_data), stepv), minor=True)
        axis.legend()
        axis.grid(True, which='both', axis='both', linestyle='--')

    def on_epoch_end(self, epoch_number: int, logs: Dict):
        """ on_epoch_end is a method invoked in the end of each epoch. """

        self.report.append(logs)

        epoch = [item['epoch'] for item in self.report]
        train_loss = [item['loss'] for item in self.report]
        val_loss = [item['val_loss'] for item in self.report]
        train_acc = [item['acc'] for item in self.report]
        val_acc = [item['val_acc'] for item in self.report]

        canvas = FigureCanvasAgg(self.fig)
        # Loss
        if train_loss is not None and val_loss is not None: 
            self._plot(self.loss_ax, epoch, train_loss, val_loss, 'Loss')
        # Accuracy
        if train_acc is not None and val_acc is not None:
            self._plot(self.acc_ax, epoch, train_acc,
                    val_acc, 'Accuracy', 100, 0, 5)

        # plt.show()
        # plt.pause(0.05)

        canvas.draw()
        fimg = np.array(canvas.renderer.buffer_rgba())
        self.fig_list.append(fimg)

    def on_train_end(self, logs: Dict):
        """ on_train_end is the method invoked in the end of each training. """
        
        if len(self.fig_list) > 0:
            imageio.mimsave(os.path.join(self.result_dir, 'result.gif'), self.fig_list, 'GIF')
            imageio.imwrite(os.path.join(self.result_dir, 'result.png'), self.fig_list[-1])
