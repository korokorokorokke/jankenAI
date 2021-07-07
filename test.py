#Import
import numpy as np
np.set_printoptions(suppress=True)

#テスト
#学習で使わなかったデータで、未知の手に対応
def test(X_test,Y_test,result,AImodel):
    successCout = 0
    count = 0

    print("    グー\tチョキ\t   パー\t\t   答え\t\t 結果   ------テスト")
    for i in range(len(X_test)):
        count += 1
        result = AImodel.predict_proba(np.array([X_test[i]]))
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