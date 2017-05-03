from six.moves import urllib
import gzip
import os
import shutil

SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'

fns = ['t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz',
       'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz']

if not os.path.exists('data'):
    os.mkdir('data')

for fn in fns:
    temp_file_name, _ = urllib.request.urlretrieve(SOURCE_URL + fn)
    shutil.copyfile(temp_file_name, os.path.join('data', fn))
