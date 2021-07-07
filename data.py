#Import
import numpy as np
np.set_printoptions(suppress=True)

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
    return (X,Y)

#1,2,3の数字を変換
def ConvertToList(target):
        if target == 1: return [1,0,0]   #1なら1,0,0
        elif target == 2: return [0,1,0] #2なら0,1,0
        elif target == 3: return [0,0,1] #3なら0,0,1

#おかしい文字入ってるか判定
def inputCheck(inp):
    for i in inp:
        if i == "1" or i == "2" or i == "3":
            pass
        else:
            return False
    return True