import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle


def save_training_history(history_file, history):
    with open(history_file, 'wb') as file_pickle:
        pickle.dump(history.history, file_pickle)


def load_training_history(history_file):
    with open(history_file, 'rb') as file_pickle:
        history = pickle.load(file_pickle)
    return history


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def title_plots(imgs, titles=None, titles_idx=None):

    qty = 5
    unique_img = is_unique_img_list(imgs)

    if unique_img:
        iterations, last_imgs = 0, 1
    else:
        iterations = len(imgs) // qty
        last_imgs = len(imgs) % qty

    for i in range(iterations):
        section_start, section_end = i * qty, (i + 1) * qty
        plots(imgs[section_start:section_end], titles=titles,
              titles_idx=titles_idx[section_start:section_end])
    if last_imgs:
        factor = qty * iterations
        img_list = imgs if unique_img else imgs[factor:factor + last_imgs + 1]
        titles_idx_list = titles_idx[0] if unique_img else titles_idx[factor:factor + last_imgs + 1]
        plots(img_list, titles=titles, titles_idx=titles_idx_list)


def plots(ims, figsize=(120, 120), rows=1, interp=False, titles=None, titles_idx=None, unique_img=None, rgb=None):

    if type(ims[0]) is np.ndarray:

        # remove 1-length sized dimensions
        ims = np.squeeze(ims)
        n_dimensions = len(ims.shape)

        if n_dimensions == 4:
            unique_img = False
            rgb = True
        elif n_dimensions == 2:
            unique_img = True
            rgb = False
        elif n_dimensions == 3:
            unique_img = ims.shape[-1] == 3
            rgb = ims.shape[-1] == 3

        # float imgs -> [0 - 255]
        if ims.dtype != np.dtype('uint8'):
            ims = img_float_to_uint8(ims)

        # preventing non-float imgs from having the wrong range
        ims = np.array(ims).astype(np.uint8)

        # channel dimension can be on 2nd dimension
        if n_dimensions == 4 and (ims.shape[-1] != 3):
            ims = ims.transpose((0, 2, 3, 1))

    f = plt.figure(figsize=figsize)
    if unique_img:
        cols = 1
    else:
        cols = len(ims) // rows if len(ims) % 2 == 0 else len(ims) // rows + 1

    it_range = 1 if unique_img else len(ims)

    for i in range(it_range):
        sp = f.add_subplot(rows, cols, i + 1)
        sp.axis('Off')

        img = ims if unique_img else ims[i]

        if titles is not None:
            sp.set_title(
                titles[np.where(titles_idx[i] == 1.)[0][0]], fontsize=200)

        if rgb:
            plt.imshow(img, interpolation=None if interp else 'none')
        else:
            plt.imshow(img, interpolation=None if interp else 'none',
                       cmap=plt.get_cmap('gray'))


def int_to_one_hot(values):
    n_values = np.max(values) + 1
    return np.eye(n_values)[values]


def confidence_to_one_hot(values):
    return int_to_one_hot(values.argmax(axis=-1))


def img_float_to_uint8(np_ar):
    old_min = np.amin(np_ar)
    old_max = np.amax(np_ar)

    if old_min < 0.0:
        old_range = old_max - old_min

        return (((np_ar - old_min) / old_range) * 255).astype('uint8')
    elif old_min >= 0.0 and old_max <= 1.0:
        return (np_ar * 255).astype('uint8')
    else:
        return np_ar.astype('uint8')


def is_unique_img_list(ims):
    ims = np.squeeze(ims)
    n_dimensions = len(ims.shape)

    if n_dimensions == 4:
        unique_img = False
    elif n_dimensions == 2:
        unique_img = True
    elif n_dimensions == 3:
        unique_img = ims.shape[-1] == 3
    return unique_img


def is_rgb_img_list(ims):
    ims = np.squeeze(ims)
    n_dimensions = len(ims.shape)

    if n_dimensions == 4:
        rgb = True
    elif n_dimensions == 2:
        rgb = False
    elif n_dimensions == 3:
        rgb = ims.shape[-1] == 3
    return rgb


# def title_plots(imgs, titles=None, titles_idx=None):
#     qty = 5

#     iterations = len(imgs) // qty
#     last_imgs = len(imgs) % qty

#     for i in range(iterations):
#         section_start, section_end = i * qty, (i + 1) * qty
#         plots(imgs[section_start:section_end], titles=titles, titles_idx=titles_idx[section_start:section_end])
#     if last_imgs:
#         plots(imgs[(i + 1) * qty:], titles=titles, titles_idx=titles_idx[(img + 1) * qty:])


# def plots(ims, figsize=(120, 120), rows=1, interp=False, titles=None, titles_idx=None):
#     if type(ims[0]) is np.ndarray:
#         ims = np.array(ims).astype(np.uint8)
#         if (ims.shape[-1] != 3):
#             ims = ims.transpose((0, 2, 3, 1))
#     f = plt.figure(figsize=figsize)
#     cols = len(ims) // rows if len(ims) % 2 == 0 else len(ims) // rows + 1
#     for i in range(len(ims)):
#         sp = f.add_subplot(rows, cols, i + 1)
#         sp.axis('Off')
#         if titles is not None:
#             sp.set_title(
#                 titles[np.where(titles_idx[i] == 1.)[0][0]], fontsize=50)
#         plt.imshow(ims[i], interpolation=None if interp else 'none')


def plot_history(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()
