#インポート
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras import optimizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)

#1,2,3の数字を変換
def ConvertToList(target):
        if target == 1: return [1,0,0]   #1なら1,0,0
        elif target == 2: return [0,1,0] #2なら0,1,0
        elif target == 3: return [0,0,1] #3なら0,0,1

#データ作成
# X   [1,0,0,2,3](0-2:前回   3:前々回   4:前々回の前)
# Y   [0,1,0](答え)
def createData(result):
    X_list = []
    Y_list = []
    for i in range(len(result) - 3):
        if i == 0:
            continue
        output1 = []
        output1.extend(ConvertToList(int(result[i])))
        output1.extend(ConvertToList(int(result[i+1])))
        output1.extend(ConvertToList(int(result[i+2])))
        X_list.append(output1)
        output2 = []
        output2.extend(ConvertToList(int(result[i+3])))
        Y_list.append(output2)
    # kerasのmodelに渡す前にX,Yをnumpyのarrayに変換する。
    X = np.array(X_list)
    Y = np.array(Y_list)

    #テストデータと訓練データに分ける
    return X,Y

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
    model.add(Dropout(0.2))

    model.add(Dense(hidden_layer_2,activation=ACTIVATION_RELU))                     #中間層-2 活性化関数はrelu ドロップアウト率は0.1(10%)
    model.add(Dropout(0.2))

    model.add(Dense(hidden_layer_3,activation=ACTIVATION_RELU))                     #中間層-3 活性化関数はrelu ドロップアウト率は0.1(10%)
    model.add(Dropout(0.2))

    model.add(Dense(output_layer,activation=ACTIVATION_SOFTMAX))                    #出力層 活性化関数はsoftmax こうすると出力三つの合計が1になる

    # モデルをコンパイル 損失関数は二値交差エントロピー 最適化関数は確率的勾配降下法
    model.compile(loss="binary_crossentropy", optimizer=optimizers.SGD(), metrics=["accuracy"])

    return model

#テスト
#学習で使わなかったデータで、未知の手に対応
def test(X_test,Y_test,result):
    successCout = 0
    count = 0

    print("    グー\tチョキ\t   パー\t\t   答え\t\t 結果   ------テスト")
    for i in range(len(X_test)):
        count += 1
        result = model.predict_proba(np.array([X_test[i]]))
        index = np.argmax(result[0])
        hand = ""
        how = "mistake"
        if index == 0: hand = "グー"
        elif index == 1: hand = "チョキ"
        elif index == 2: hand = "パー"

        if Y_test[i][0] == 1 and hand == "グー": how="accurate"
        elif Y_test[i][1] == 1 and hand == "チョキ": how="accurate"
        elif Y_test[i][2] == 1 and hand == "パー": how="accurate"

        if how == "accurate": successCout += 1
        print(result,"\t",Y_test[i],"\t",how)
    print("aiの予測の的中率 :",successCout/count*100 if not count == 0 else 0)

#おかしい文字入ってるか判定
def inputCheck(inp):
    for i in inp:
        if i == "1" or i == "2" or i == "3":
            pass
        else:
            return False
    return True

print("じゃんけんの手を入力してください\r\n古い->新しい,1=グー 2=チョキ 3=パーを区切りなしで\r\n50個数以上登録してください\r\n1,2,3意外を入力すると塩見\r\nsを入力するとさざえ")
inp = input()

#さざえ
#1232113213123122312312311231321231133123121332213212231332131213312121311232331212233211112312132231123113223132113121233211332231213321232132231123132113321232131321312232133213212332112321331221323312231132312213232113233122131233211232312122313322131123211323311221323132121332123312321132213121332123221321231132321331221332132213231123213232113231233221323112323132213123211321211332131213332132122231131322331211223112331231221231211231222332231312233113223231322323112221232123113231311223132321132332212321132321322312311213122132312231232213211312112231123123312231323213221223311123233121332333211132133221233132233212113323221312311231323321311231122311112321233231132322131221333121213223221123213213233112322132311123231323321233121232231323221331213121132231121232211213321133122131122332131232112323132211232132321323113221323212132123213212312133222321231223312123311231332123112323123133212312231223311232132133122311232313121232132321321132331223112332133121321312231233112331213212133213121231123321332112331213321332123132132322113321331223113232123211321332121323233211231232311233212332113223123322131232113231211323132312321223132312332112313212312213321231212311231321233212231132123121332113213321213221331231122331321312232112323113212321133213223113221322133213212311

#しおみ
#1132123123312231223113223123311232123113232112322131231332213123133221132133222312123312311233121321231132212332123132113212132132123312213212313211233213232212321323132311213213123123122332

if inputCheck(inp):
    history = inp
elif inp=="s":
    history = str(1232113213123122312312311231321231133123121332213212231332131213312121311232331212233211112312132231123113223132113121233211332231213321232132231123132113321232131321312232133213212332112321331221323312231132312213232113233122131233211232312122313322131123211323311221323132121332123312321132213121332123221321231132321331221332132213231123213232113231233221323112323132213123211321211332131213332132122231131322331211223112331231221231211231222332231312233113223231322323112221232123113231311223132321132332212321132321322312311213122132312231232213211312112231123123312231323213221223311123233121332333211132133221233132233212113323221312311231323321311231122311112321233231132322131221333121213223221123213213233112322132311123231323321233121232231323221331213121132231121232211213321133122131122332131232112323132211232132321323113221323212132123213212312133222321231223312123311231332123112323123133212312231223311232132133122311232313121232132321321132331223112332133121321312231233112331213212133213121231123321332112331213321332123132132322113321331223113232123211321332121323233211231232311233212332113223123322131232113231211323132312321223132312332112313212312213321231212311231321233212231132123121332113213321213221331231122331321312232112323113212321133213223113221322133213212311)
else:
    history = str(1132123123312231223113223123311232123113232112322131231332213123133221132133222312123312311233121321231132212332123132113212132132123312213212313211233213232212321323132311213213123123122332)#古い->新しい

sepIndex = len(history) - int(len(history)/10)

#上の1,2,3の羅列から、AI用のデータに変換
(X,Y) = createData(history[0:sepIndex])

#上で使わなかった部分をテスト用にする
(X_test,Y_test) = createData(history[sepIndex:-1])

#学習回数
epochs = 500
#さざえ: 128,256,64 1000epochs
#S: 64,128,32  1000epochs

#AI作成
model = createModel(9,128,256,64,3)

#AI学習開始
result = model.fit(X, Y, nb_epoch=epochs, batch_size=int(len(X)/5))# 学習を実行

#AIの形を出力
model.summary()

#テスト(未知の手に対応できるか)
test(X_test,Y_test,result)

#最後2つから予測(答えはないだろw)
targetList = []
ta = history[-3:]
for i in ta:
    if i == "1": targetList.extend([1,0,0])
    elif i == "2": targetList.extend([0,1,0])
    elif i == "3": targetList.extend([0,0,1])

print("最後の3つから次を予測 :",model.predict_proba(np.array([targetList])))