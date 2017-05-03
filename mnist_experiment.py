import numpy as np
import mnist
import fnn
import check_grad as cg
import matplotlib.pyplot as plt

def preprocess(data):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = (data - data.mean(axis=0)) / data.std(axis=0)
    result[np.isnan(result)] = 0.0
    return result


def create_one_hot_labels(labels, dim=10):
    one_hot_labels = np.zeros((labels.shape[0], dim))
    for i in xrange(labels.shape[0]):
        one_hot_labels[i][labels[i]] = 1
    return one_hot_labels


def data_preprocessing(data_dir, seed=None):
    # Load mnist data
    mn = mnist.MNIST(data_dir)

    mnist_test_data, mnist_test_labels = mn.load_testing()
    mnist_train_data, mnist_train_labels = mn.load_training()
    raw_test_data = np.array(mnist_test_data)
    raw_train_data = np.array(mnist_train_data)

    # Convert into matrix and one hot vector and preprocess to
    # have mean=0.0 and std=1.0    
    mnist_test_data = preprocess(raw_test_data)
    mnist_train_data = preprocess(raw_train_data)
    mnist_test_labels = create_one_hot_labels(np.array(mnist_test_labels))
    mnist_train_labels = create_one_hot_labels(np.array(mnist_train_labels))

    # Split into training and validation set.
    if seed is not None:
        np.random.seed(seed)
    n = mnist_train_data.shape[0]
    indices = np.random.permutation(n)
    n_train = int((55.0/60)*n)
    train_idx, valid_idx = indices[:n_train], indices[n_train:]
    train_data, train_labels = mnist_train_data[train_idx,:], mnist_train_labels[train_idx,:]
    valid_data, valid_labels = mnist_train_data[valid_idx,:], mnist_train_labels[valid_idx,:]

    # Get test set.
    test_data, test_labels = mnist_test_data, mnist_test_labels

    return (train_data, train_labels, valid_data, valid_labels,
            test_data, test_labels, raw_train_data, raw_test_data)
    
    
def main():
    print 'Loading and preprocessing data\n'
    (train_data, train_labels, valid_data,
     valid_labels, test_data, test_labels,
     raw_train_data, raw_test_data) = data_preprocessing('data/')

    # Initialize model
    print 'Initializing neural network\n'
    model = fnn.FNN(784, 10, [128, 32], [fnn.relu, fnn.relu])

    selected = np.random.randint(test_data.shape[0], size=100)
    true_labels = np.argmax(test_labels[selected], axis=1)
    preds_init = model.predict(test_data[selected])

    print 'Start training\n'
    n_train = train_data.shape[0]
    n_epochs = 50
    batch_size = 100
    opt = fnn.GradientDescentOptimizer(0.01)
    for i in xrange(n_epochs):
        sum_loss = 0.0
        for j in xrange((n_train - 1) // batch_size + 1):
            batch_data = train_data[j*batch_size:(j+1)*batch_size]
            batch_labels = train_labels[j*batch_size:(j+1)*batch_size]
            _, loss = model.forwardprop(batch_data, batch_labels)
            if np.isnan(loss):
                print 'batch %s loss is abnormal' % j
                print loss
                continue
            sum_loss += loss
            model.backprop(batch_labels)
            opt.update(model)
        train_loss = sum_loss/(j+1)
        _, valid_loss = model.forwardprop(valid_data, valid_labels)

        train_accuracy = (np.sum(model.predict(train_data) == 
                                 np.argmax(train_labels, axis=1)) / 
                          np.float(train_labels.shape[0]))
        valid_accuracy = (np.sum(model.predict(valid_data) == 
                                 np.argmax(valid_labels, axis=1)) / 
                          np.float(valid_labels.shape[0]))
        print '=' * 20 + ('Epoch %d' % i) + '=' * 20
        print('Train loss %s accuracy %s\nValid loss %s accuracy %s\n' % 
              (train_loss, train_accuracy, valid_loss, valid_accuracy))

    # Compute test loss and accuracy.
    _, test_loss = model.forwardprop(test_data, test_labels)
    test_accuracy = (np.sum(model.predict(test_data) == 
                            np.argmax(test_labels, axis=1)) / 
                     np.float(test_labels.shape[0]))
    print '=' * 20 + 'Training finished' + '=' * 20 + '\n'
    print ('Test loss %s accuracy %s\n' %
           (test_loss, test_accuracy))

    preds_trained = model.predict(test_data[selected])

    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    fig.subplots_adjust(wspace=0)
    for a, image, true_label, pred_init, pred_trained in zip(
            axes.flatten(), raw_test_data[selected],
            true_labels, preds_init, preds_trained):
        a.imshow(image.reshape(28, 28), cmap='gray_r')
        a.text(0, 10, str(true_label), color="black", size=15)
        a.text(0, 26, str(pred_trained), color="blue", size=15)
        a.text(22, 26, str(pred_init), color="red", size=15)

        a.set_xticks(())
        a.set_yticks(())

    plt.show()
    

if __name__ == '__main__':
    main()
