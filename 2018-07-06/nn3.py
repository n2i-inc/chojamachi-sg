import cv2
import numpy as np


#必要な関数の定義
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def nn3(filepath):
    #画像の読み取り
    img = cv2.imread(filepath)
    img = np.resize(img, (1, 28, 28))
    img = img.flatten()
    img = np.reshape(img, (-1, 784))
    

    #重みの設定
    w1 = np.random.randn(784, 100)
    w2 = np.random.randn(100, 50)
    w3 = np.random.randn(50, 10)

    #neuralnet
    h1 = np.dot(img, w1)
    y1 = relu(h1)
    h2 = np.dot(y1, w2)
    y2 = relu(h2)
    h3 = np.dot(y2, w3)
    y = softmax(h3)

    #nnの結果を出力
    print('{}'.format(np.argmax(y, axis = 1)))

if __name__ == '__main__':
    filepath = '画像ファイルのパス'
    nn3(filepath)