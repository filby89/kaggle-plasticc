import torch
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [16, 9]

def accuracy(output, target, topk=(1,), weighted = False):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # print(pred)
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

astro_mapper = {
    6: 0,
   15: 1, 16:2, 42:3, 52:4, 53:5, 62:6, 64:7, 65:8, 67:9, 88:10, 90:11, 92:12, 95:13, None: None
}

def get_weighted_loss_weights(dataset, num_classes):
    print("Calculating sampler weights...")
    labels_array = [astro_mapper[x.label] for x in dataset]
    # labels_array = dataset#.Y_body

    from sklearn.utils import class_weight
    import numpy as np
    class_weights = class_weight.compute_class_weight('balanced', np.unique(labels_array), labels_array)
    assert(class_weights.size == num_classes)
    class_weights = 1/class_weights
    print("Class Weights: ", class_weights)
    return class_weights


# calculates the weights for doing balanced sampling
def get_sampler_weights(dataset, num_classes):
    print("Calculating sampler weights...")
    labels_array = [astro_mapper[x.label] for x in dataset]

    from sklearn.utils import class_weight
    import numpy as np
    class_weights = class_weight.compute_class_weight('balanced', np.unique(labels_array), labels_array)
    assert(class_weights.size == num_classes)

    sampler_weights = torch.zeros(len(labels_array))
    i=0
    for label in labels_array:
        sampler_weights[i] = class_weights[int(label)]
        # print(i)
        i+=1

    return sampler_weights


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def load_checkpoint(checkpoint_file):
	return torch.load(checkpoint_file)


def save_checkpoint(state, filename):
	filename = 'checkpoints/%s'%filename
	torch.save(state, filename)



def pad_sequence(sequences, batch_first=False, padding_value=0, max_len=100):
    r"""Pad a list of variable length Tensors with zero

    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is list of
    sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
    otherwise.

    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *`` where `T` is the
            length of the longest sequence.
        Function assumes trailing dimensions and type of all the Tensors
            in sequences are same.

    Arguments:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.

    Returns:
        Tensor of size ``T x B x *`` if batch_first is False
        Tensor of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    # max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor

import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, filename=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('%s.png'%filename)


def calc_gradients(params):
    grad_array = []
    _mean = []
    _max = []
    for param in params:
        grad_array.append(param.grad.data)
        _mean.append(torch.mean(param.grad.data))
        _max.append(torch.max(param.grad.data))
    print(np.mean(_mean))
    print(np.max(_max))


def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

import errno
import os


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse
from sklearn.utils import check_array
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.validation import (check_is_fitted, check_random_state,
                                FLOAT_DTYPES)

class TimeSeriesMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, *args):
        self.min_max_scaler = StandardScaler(*args)

    def fit(self, X, y=None):
        self.min_max_scaler._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        self.min_max_scaler.partial_fit(X.reshape((-1, X.shape[2])))


    def transform(self, X):
        X = [self.min_max_scaler.transform(x) for x in X]
        return np.array(X)

    def inverse_transform(self, X):
        X = [self.min_max_scaler.inverse_transform(x) for x in X]
        return np.array(X)
