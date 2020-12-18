import os
import numpy as np
from scipy import ndimage
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle


def gaussian_k(x0, y0, sigma, width, height):
    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)[:, np.newaxis]
    return np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))

def get_y_as_heatmap(y, width, height, sigma=3):
    ymap = []
    for i in range(y.shape[0]):
        msk_array = np.zeros((width, height, 15), dtype=np.float32)
        for j in range(15):
            msk = gaussian_k(y[i, 2*j]*width, y[i, 2*j+1]*height, sigma, width, height)
            msk_array[:,:,j] = msk
        ymap.append(msk_array)
    ymap = np.array(ymap)
    return ymap

def img_affine(x, y, dx, dy, width, height):
    xt = ndimage.shift(x, (dy, dx), mode='nearest')
    yt = np.zeros(30)
    for i in range(15):
        yt[2*i] = y[2*i] + dx / width
        yt[2*i+1] = y[2*i+1] + dy / height
    return xt, yt

def img_hflip(x, y):
    xf = np.fliplr(x)
    yf = np.zeros(30)
    for i in range(15):
        yf[2*i] = 1.0 - y[2*i]
        yf[2*i+1] = y[2*i+1]
    return xf, yf

def plot_sample(X,y,axs):
    '''
    kaggle picture is 96 by 96
    y is rescaled to range between -1 and 1
    '''
    axs.imshow(X.reshape(96,96),cmap="gray")
    axs.scatter(48*y[0::2]+ 48,48*y[1::2]+ 48)
    
def load(fname, test=False, cols=None):
    """
    load test/train data
    cols : a list containing landmark label names.
           If this is specified, only the subset of the landmark labels are 
           extracted. for example, cols could be:
           [left_eye_center_x, left_eye_center_y]        
    return: 
    X: 2-d numpy array (Nsample, Ncol*Nrow)
    y: 2-d numpy array (Nsample, Nlandmarks*2) 
       In total there are 15 landmarks. 
       As x and y coordinates are recorded, u.shape = (Nsample,30)
    """
    df = read_csv(os.path.expanduser(fname)) 
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    if cols:  
        df = df[list(cols) + ['Image']]

    myprint = df.count()
    myprint = myprint.reset_index()
    # print(myprint)  
    ## row with at least one NA columns are removed!
    df = df.dropna()  
    X = np.vstack(df['Image'].values) / 255.  # changes valeus between 0 and 1
    X = X.astype(np.float32)
    if not test:  # labels only exists for the training data
        ## standardization of the response
        y = df[df.columns[:-1]].values
        # y = (y - 48) / 48  # y values are between [-1,1]
        y /= 96 # y values are between [0,1]
        X, y = shuffle(X, y, random_state=42)  # shuffle data
        y = y.astype(np.float32)
    else:
        y = None 
    return X, y

def get_avg_xy(msk, n_points):
    h,w = msk.shape
    # 
    idx = np.argsort(msk, axis=None)
    idx = idx[-n_points:]
    hms = msk.flatten()[idx]
    x = idx % w
    y = np.floor(idx / h)
    cx = np.sum(x * hms) / np.sum(hms)
    cy = np.sum(y * hms) / np.sum(hms)

    return np.array([cy, cx]), msk[int(np.round(cy)), int(np.round(cx))]