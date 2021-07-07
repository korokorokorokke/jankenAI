#Import
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
import numpy as np
np.set_printoptions(suppress=True)

#入力層  :5 (入力が5だから)
#中間層-1:13
#中間層-2:30
#中間層-3:10
#出力層  :3 (出力が3だから)
def createModel(input_layer,hidden_layer_1,hidden_layer_2,hidden_layer_3,output_layer):
    ACTIVATION_RELU = "relu"
    ACTIVATION_SOFTMAX = "softmax"

    model = Sequential()# モデルを生成する

    model.add(Dense(input_layer,input_dim=input_layer,activation=ACTIVATION_RELU))  #入力層 活性化関数はrelu

    model.add(Dense(hidden_layer_1,activation=ACTIVATION_RELU))                     #中間層-1 活性化関数はrelu ドロップアウト率は0.1(10%)
    model.add(Dropout(0.1))

    model.add(Dense(hidden_layer_2,activation=ACTIVATION_RELU))                     #中間層-2 活性化関数はrelu ドロップアウト率は0.1(10%)
    model.add(Dropout(0.1))

    model.add(Dense(hidden_layer_3,activation=ACTIVATION_RELU))                     #中間層-3 活性化関数はrelu ドロップアウト率は0.1(10%)
    model.add(Dropout(0.1))

    model.add(Dense(output_layer,activation=ACTIVATION_SOFTMAX))                    #出力層 活性化関数はsoftmax こうすると出力三つの合計が1になる

    # モデルをコンパイル 損失関数は二値交差エントロピー 最適化関数は確率的勾配降下法
    model.compile(loss="binary_crossentropy", optimizer=optimizers.SGD(), metrics=["accuracy"])

    return model