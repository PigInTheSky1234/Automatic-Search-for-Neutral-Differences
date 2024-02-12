import numpy as np
import stp
import time
import numpy as np
import math as mt
import BCT_UBCT_LBCT_EBCT as Matrice
import os
from datetime import datetime
import lea

np.set_printoptions(threshold=np.inf)


def ReverseSKOneRound(SK, R, Rotl_Cons=[1, 3, 6, 11, 13, 17]):
    NumWord_SK = len(SK)
  
    if NumWord_SK <= 6:
        OperationIndex = range(NumWord_SK)
    else:
        OperationIndex = [(6 * R) % 8,
                          (6 * R + 1) % 8,
                          (6 * R + 2) % 8,
                          (6 * R + 3) % 8,
                          (6 * R + 4) % 8,
                          (6 * R + 5) % 8,
                          ]
   
    Temp_SK = SK.copy()
    for i in range(len(OperationIndex)):
        Temp_SK[OperationIndex[i]] = lea.ror(Temp_SK[OperationIndex[i]], Rotl_Cons[i])

    Delta = np.squeeze(lea.delta)

    cons = Delta[R % NumWord_SK]

    for i in range(len(OperationIndex)):
        cons_Used = lea.rol(cons, R + i)
        Temp_SK[OperationIndex[i]] -= cons_Used

        if Temp_SK[OperationIndex[i]] < 0:
            Temp_SK[OperationIndex[i]] += 2 ** 32
    return Temp_SK


class LEA():
    def __init__(self, nrounds, m=4, WordSize=32, BoundDiffPropagation={}, RelatedKey=False):

        self._nrounds = nrounds  
        self._m = m  
        self._WordSize = WordSize
 
        self.RelatedKey = RelatedKey

       
        self._Alpha, self._Beta, self._Gamma = 9, 5, 3
    
        self._SK_RotrNum = [1, 3, 6, 11, 13, 17]
        self._SK128_UsingSKIndex = [0, 1, 2, 1, 3, 1]
        self._SK_Cons = [0xc3efe9db, 0x44626b02, 0x79e27c8a, 0x78df30ec, 0x715ea49e, 0xc785da0a, 0xe04ef22a, 0xe5c40957]

        self._NumWord = 4
        self._NumAddition = 3

   
        self._Solver = stp.Solver()

        self._P = [self.GetP(i) for i in range(nrounds[0] + nrounds[1] + nrounds[2] + 1)]
   
        self._Pr = [self.GetPr(i) for i in range(nrounds[0] + nrounds[1] + nrounds[2])]


        self._K = [self.GetK(i) for i in
                   range(nrounds[0] + self._m - 2 + max(0, nrounds[1] - 1))] 
        self._K_Pr = [self.GetPr(i, VarName="k_pr") for i in range(nrounds[0] - 1 + max(0, nrounds[1] - 1))]

 
        if (len(BoundDiffPropagation) == 0):
            self._BoundDiffPropagation = {
                "LEA128": [0, 0, 0, 2, 6],

            }
        else:
            self._BoundDiffPropagation = BoundDiffPropagation

    def GetP(self, R, ):

        r0 = self._nrounds[0]
        r1 = self._nrounds[1]
        r2 = self._nrounds[2]
        if (R < r0) | (R > (r0 + r1)):
            P = [self._Solver.bitvec("p" + str(R) + "_" + str(i), self._WordSize) for i in range(self._NumWord)]
        elif r1 == 0:
            P = [self._Solver.bitvec("p" + str(R) + "_" + str(i), self._WordSize) for i in range(self._NumWord)]
        else:

            P = [self._Solver.bitvec("p" + str(R) + "_" + str(i), self._WordSize) for i in range(self._NumWord * 2)
                 ]
        return P

    def GetStates(self, prefix, nr, NumWord, OuputVarorStr=True):
  
        if OuputVarorStr:
            S = [
                [self._Solver.bitvec(prefix + str(R) + "_" + str(i), self._WordSize) for i in range(NumWord)]
                for R in range(nr)
            ]
        else:
            S = [
                [prefix + str(R) + "_" + str(i) for i in range(NumWord)]
                for R in range(nr)
            ]

        return S

    def GetPr(self, R, VarName="pr", suffix=""):
   
        Pr = [self._Solver.bitvec(VarName + str(R) + "_" + str(i) + suffix, self._WordSize) for i in
              range(self._NumAddition)
              ]
        return Pr

    def GetK(self, R, VarName="k"):
   
        r0 = self._nrounds[0]
        r1 = self._nrounds[1]
        if (R < (r0 + max(r1 - 1, 0))):  
            K = [self._Solver.bitvec(VarName + str(R) + "L", self._WordSize),
                 self._Solver.bitvec(VarName + str(R) + "R", self._WordSize), ]
        else:
            K = [self._Solver.bitvec(VarName + str(R) + "L", self._WordSize), ]

        return K

    def rotr(self, X, n, r):
    
   
        Mask1 = 2 ** n - 1
        Mask2 = 2 ** r - 1
        Mask3 = Mask1 - Mask2
        if not 0 <= r < n:
            raise ValueError("r must be in the range 0 <= r < %d" % n)

 

        return ((X) << (n - r)) | ((X) >> r)

    def rotl(self, X, n, r):
      
        return self.rotr(X, n, n - r)

    def CountOnes(self, X, NumLen):
        Counter = X.extract(0, 0)
        for i in range(1, NumLen):
            Counter = Counter + X.extract(i, i)
        return Counter

    def HW(self, x, NumLen, StartIndex=0):
        return sum((x & (1 << i)) >> i for i in range(StartIndex, NumLen))

    def HWList(self, x, NumLen, StartIndex=0):
        return [(x & (1 << i)) >> i for i in range(StartIndex, NumLen)]

    def Cons_Addition_Diff(self, Alpha, Beta, Gamma, Pr):

        n = self._WordSize

        Constraints = []
        x1, x2, y0 = Alpha, Beta, Gamma

        AllOne = 2 ** self._WordSize - 2

        x1_Rotl_1 = self.rotl(x1, n, 1)
        x2_Rotl_1 = self.rotl(x2, n, 1)
        y0_Rotl_1 = self.rotl(y0, n, 1)

        Constraints += [AllOne == (AllOne & (x1 | x2 | y0.not_() | x1_Rotl_1 | x2_Rotl_1 | y0_Rotl_1))]
        Constraints += [AllOne == (AllOne) & (
                x1 | x2.not_() | y0 | x1_Rotl_1 | x2_Rotl_1 | y0_Rotl_1)]
        Constraints += [AllOne == (AllOne) & (
                x1.not_() | x2 | y0 | x1_Rotl_1 | x2_Rotl_1 | y0_Rotl_1)]
        Constraints += [AllOne == (AllOne) & (
                x1.not_() | x2.not_() | y0.not_() | x1_Rotl_1 | x2_Rotl_1 | y0_Rotl_1)]

        Constraints += [AllOne == (AllOne) & (
                x1 | x2 | y0 | x1_Rotl_1.not_() | x2_Rotl_1.not_() | y0_Rotl_1.not_())]
        Constraints += [AllOne == (AllOne) & (
                x1 | x2.not_() | y0.not_() | x1_Rotl_1.not_() | x2_Rotl_1.not_() | y0_Rotl_1.not_())]
        Constraints += [AllOne == (AllOne) & (
                x1.not_() | x2 | y0.not_() | x1_Rotl_1.not_() | x2_Rotl_1.not_() | y0_Rotl_1.not_())]
        Constraints += [AllOne == (AllOne) & (
                x1.not_() | x2.not_() | y0 | x1_Rotl_1.not_() | x2_Rotl_1.not_() | y0_Rotl_1.not_())]

        Constraints += [0 == (1 & x1 ^ 1 & x2 ^ 1 & y0)]


        Constraints += [AllOne == AllOne & (x1_Rotl_1.not_() | y0_Rotl_1 | Pr)]
        Constraints += [AllOne == AllOne & (x2_Rotl_1 | y0_Rotl_1.not_() | Pr)]
        Constraints += [AllOne == AllOne & (x1_Rotl_1 | x2_Rotl_1.not_() | Pr)]

        Constraints += [AllOne == AllOne & (x1_Rotl_1 | x2_Rotl_1 | y0_Rotl_1 | Pr.not_())]
        Constraints += [AllOne == AllOne & (x1_Rotl_1.not_() | x2_Rotl_1.not_() | y0_Rotl_1.not_() | Pr.not_())]
        Constraints += [(Pr & 1) == 0]

        return Constraints

    def RoundCons_DiffPropagation(self, R, Pr2=[], NBSearch=False, RelatedKey=False, StartRound=-1):
        Constraints = []
        r0, r1, r2 = self._nrounds[0], self._nrounds[1], self._nrounds[2]


        InputD = self._P[R][:self._NumWord]
        OutputD = self._P[R + 1][:self._NumWord]
        if len(Pr2) > 0:
            assert StartRound >= 0, "如果使用Pr2，必须有StartRound>=0!!"
            Pr = Pr2[R - StartRound]
        else:
            Pr = self._Pr[R]
        n = self._WordSize
 
        Constraints += [OutputD[-1] == InputD[0]]

    
        Constraints += self.Cons_Addition_Diff(InputD[0], InputD[1], self.rotr(OutputD[0], n, self._Alpha), Pr[0])
    
        Constraints += self.Cons_Addition_Diff(InputD[1], InputD[2], self.rotl(OutputD[1], n, self._Beta), Pr[1])
   
        Constraints += self.Cons_Addition_Diff(InputD[2], InputD[3], self.rotl(OutputD[2], n, self._Gamma), Pr[2])

        return Constraints

    def RoundCons_DiffPropagationOfK(self, R, ):
        Constraints = []

        n = self._WordSize
        Pr = self._K_Pr[R]

        X = self._K[R][:2]
        x0 = X[0]
        x1 = X[1]
        y0 = self._K[R + self._m - 1][0]
        y1 = self._K[R + 1][1]

        x2 = self.rotr(x0, n, self._Alpha)
        x3 = self.rotl(x1, n, self._Beta)



        Constraints = Constraints + [x3 == (y1 ^ y0)]


        AllOne = 2 ** self._WordSize - 2

        x1_Rotl_1 = self.rotl(x1, n, 1)
        x2_Rotl_1 = self.rotl(x2, n, 1)
        y0_Rotl_1 = self.rotl(y0, n, 1)

        Constraints += [AllOne == (AllOne & (x1 | x2 | y0.not_() | x1_Rotl_1 | x2_Rotl_1 | y0_Rotl_1))]
        Constraints += [AllOne == (AllOne) & (
                x1 | x2.not_() | y0 | x1_Rotl_1 | x2_Rotl_1 | y0_Rotl_1)]
        Constraints += [AllOne == (AllOne) & (
                x1.not_() | x2 | y0 | x1_Rotl_1 | x2_Rotl_1 | y0_Rotl_1)]
        Constraints += [AllOne == (AllOne) & (
                x1.not_() | x2.not_() | y0.not_() | x1_Rotl_1 | x2_Rotl_1 | y0_Rotl_1)]

        Constraints += [AllOne == (AllOne) & (
                x1 | x2 | y0 | x1_Rotl_1.not_() | x2_Rotl_1.not_() | y0_Rotl_1.not_())]
        Constraints += [AllOne == (AllOne) & (
                x1 | x2.not_() | y0.not_() | x1_Rotl_1.not_() | x2_Rotl_1.not_() | y0_Rotl_1.not_())]
        Constraints += [AllOne == (AllOne) & (
                x1.not_() | x2 | y0.not_() | x1_Rotl_1.not_() | x2_Rotl_1.not_() | y0_Rotl_1.not_())]
        Constraints += [AllOne == (AllOne) & (
                x1.not_() | x2.not_() | y0 | x1_Rotl_1.not_() | x2_Rotl_1.not_() | y0_Rotl_1.not_())]

        Constraints += [0 == (1 & x1 ^ 1 & x2 ^ 1 & y0)]


        Constraints += [AllOne == AllOne & (x1_Rotl_1.not_() | y0_Rotl_1 | Pr)]
        Constraints += [AllOne == AllOne & (x2_Rotl_1 | y0_Rotl_1.not_() | Pr)]
        Constraints += [AllOne == AllOne & (x1_Rotl_1 | x2_Rotl_1.not_() | Pr)]

        Constraints += [AllOne == AllOne & (x1_Rotl_1 | x2_Rotl_1 | y0_Rotl_1 | Pr.not_())]
        Constraints += [AllOne == AllOne & (x1_Rotl_1.not_() | x2_Rotl_1.not_() | y0_Rotl_1.not_() | Pr.not_())]

        Constraints += [(Pr & 1) == 0]

        return Constraints

    def DiffPropagationOfK(self, StartRound, R, ):
        Constraints = []


        for i in range(StartRound, StartRound + R):
            Constraints += self.RoundCons_DiffPropagationOfK(i)


        return Constraints

    def DiffPropagation(self, StartRound, R, Pr=[], DiffPropa=[], NBSearch=False, RelatedKey=False):
        Constraints = []


        for i in range(StartRound, StartRound + R):
            Constraints += self.RoundCons_DiffPropagation(i, Pr, NBSearch=NBSearch, RelatedKey=RelatedKey,
                                                          StartRound=StartRound)

        
        if len(DiffPropa) != 0:

            if len(DiffPropa) == 2:  
           
                InputDiff, OutputDiff = DiffPropa[0], DiffPropa[1]

                for j in range(self._NumWord):
                    Constraints += [(self._P[StartRound][j] ^ InputDiff[j]) == 0]
                    Constraints += [(self._P[StartRound + R][j] ^ OutputDiff[j]) == 0]
            else: 
                assert len(DiffPropa) == (R + 1), "Error！"
                for r in range(R + 1):
                    for j in range(self._NumWord):
                        Constraints += [(self._P[StartRound + r][j] ^ DiffPropa[r][j]) == 0]
                        Constraints += [(self._P[StartRound + r][j] ^ DiffPropa[r][j]) == 0]

        return Constraints

    def SearchDiff(self, HWPr=0, AddMatSuiBound=False, ExcludePoint=[], SpecifyDiff=[[], []],
                   HWInDiff=-1, HWOutDiff=-1
                   ):

     
        Constraints = self.DiffPropagation(0, self._nrounds[0])
 
        Diffs = self._P[0]
        temp = 0
        for j in range(len(Diffs)):
            if j == 0:
                temp = Diffs[j]
            else:
                temp = temp | Diffs[j]

        Constraints += [temp >= 1]


        ObjList = []

        for i in range(len(self._Pr)):
            if i == 0:
                temp = []
                for j in range(self._NumAddition):
                    temp += self.HWList(self._Pr[i][j], self._WordSize)

                ObjList.append(temp)
          

            else:
                temp = []
                for j in range(self._NumAddition):
                    temp += self.HWList(self._Pr[i][j], self._WordSize)

                ObjList.append(temp + ObjList[-1])
           

        Constraints += [sum(ObjList[-1]) == HWPr]

  
        if AddMatSuiBound:

            Bound = self._BoundDiffPropagation["LEA128"]
            for i in range(len(self._Pr) - 1):
                Constraints += [sum(ObjList[i]) >= Bound[i + 1]]

     
        for i in range(len(ExcludePoint)):
            Diffs = self._P[0] + self._P[-1]

            Impossible = ExcludePoint[i]
            temp = 0
            for j in range(self._NumWord * 2):
                if j == 0:
                    temp = Diffs[j] ^ Impossible[j]
                else:
                    temp = temp | (Diffs[j] ^ Impossible[j])

            Constraints += [temp >= 1]


        if len(SpecifyDiff[0]) > 0:
            temp = 0
            for i in range(self._NumWord):
                if i == 0:
                    temp = self._P[0][i] ^ SpecifyDiff[0][i]
                else:
                    temp = temp | (self._P[0][i] ^ SpecifyDiff[0][i])

            Constraints += [temp == 0]

        if len(SpecifyDiff[1]) > 0:
            temp = 0
            for i in range(self._NumWord):
                if i == 0:
                    temp = self._P[-1][i] ^ SpecifyDiff[1][i]
                else:
                    temp = temp | (self._P[-1][i] ^ SpecifyDiff[1][i])

            Constraints += [temp == 0]


        if HWInDiff >= 0:
            Diffs = self._P[0]
            temp = []
            for j in Diffs:
                temp += self.HWList(j, self._WordSize)

            Constraints += [sum(temp) <= HWInDiff]

        if HWOutDiff >= 0:
            Diffs = self._P[-1]
            temp = []
            for j in Diffs:
                temp += self.HWList(j, self._WordSize)

            Constraints += [sum(temp) <= HWOutDiff]

  
        for i in Constraints:
            self._Solver.add(i)

        Result = self._Solver.check()

        model = {}
        if Result:  
            model = self._Solver.model()
      

        return Result, model


    def Cons_Addition_EBCT(self, D0L, D0R, D3L, D3R, D1L, D1R, D2L, D2R, HWPr, Suffix):
       
        Constraints = []


        Constraints += [D0R == D1R, D3R == D2R, ]

       
        S0 = self._Solver.bitvec("S0_R" + Suffix, self._WordSize)
        S1 = self._Solver.bitvec("S1_R" + Suffix, self._WordSize)
        S2 = self._Solver.bitvec("S2_R" + Suffix, self._WordSize)
        S3 = self._Solver.bitvec("S3_R" + Suffix, self._WordSize)
        Flag = self._Solver.bitvec("Flag_R" + Suffix, self._WordSize)


        Constraints += [
            S0 & 1 == 0,
            S1 & 1 == 1,
            S2 & 1 == 0,
            S3 & 1 == 0,
        ]
   
        NS0 = self.rotr(S0, self._WordSize, 1)
        NS1 = self.rotr(S1, self._WordSize, 1)
        NS2 = self.rotr(S2, self._WordSize, 1)
        NS3 = self.rotr(S3, self._WordSize, 1)
        AllOne = 2 ** (self._WordSize - 1) - 1  
        AllOne2 = 2 ** (self._WordSize) - 1  

        Diff0, Diff1, Diff2, Diff3, Diff4, Diff5 = D0L, D0R, D1L, D2L, D2R, D3L

        Constraints += [AllOne & Flag == AllOne & (~S0 | ~S1) & (~S0 | ~S2) & (~S0 | Diff3 | ~Diff4 | ~Diff5) & (
                ~S0 | ~Diff3 | Diff4 | ~Diff5) & (~S0 | ~Diff3 | ~Diff4 | Diff5) & (
                                ~S0 | Diff3 | Diff4 | Diff5) & (Diff0 | ~Diff1 | Diff3 | ~Diff4 | ~Diff5) & (
                                ~Diff0 | Diff1 | Diff3 | ~Diff4 | ~Diff5) & (
                                Diff0 | ~Diff2 | ~Diff3 | Diff4 | ~Diff5) & (
                                Diff1 | ~Diff2 | Diff3 | Diff4 | ~Diff5) & (
                                ~Diff0 | Diff2 | ~Diff3 | Diff4 | ~Diff5) & (
                                ~Diff1 | Diff2 | Diff3 | Diff4 | ~Diff5) & (
                                Diff1 | ~Diff2 | ~Diff3 | ~Diff4 | Diff5) & (
                                Diff0 | ~Diff2 | Diff3 | ~Diff4 | Diff5) & (
                                Diff0 | ~Diff1 | ~Diff3 | Diff4 | Diff5) & (
                                ~Diff0 | Diff1 | ~Diff3 | Diff4 | Diff5) & (
                                ~Diff1 | Diff2 | ~Diff3 | ~Diff4 | Diff5) & (
                                ~Diff0 | Diff2 | Diff3 | ~Diff4 | Diff5) & (
                                ~S1 | ~Diff2 | Diff3 | ~Diff4 | ~Diff5) & (
                                ~S1 | ~Diff1 | ~Diff3 | Diff4 | ~Diff5) & (
                                ~S1 | ~Diff0 | ~Diff3 | ~Diff4 | Diff5) & (
                                ~S1 | Diff0 | Diff1 | ~Diff2 | Diff3) & (
                                ~S1 | Diff0 | ~Diff1 | Diff2 | Diff4) & (
                                ~S1 | ~Diff0 | Diff1 | Diff2 | Diff5) & (~S1 | ~S3 | Diff0 | Diff1 | ~Diff2) & (
                                ~S1 | ~S3 | Diff0 | ~Diff1 | Diff2) & (~S1 | ~S3 | ~Diff0 | Diff1 | Diff2) & (
                                ~S1 | ~S3 | ~Diff0 | ~Diff1 | ~Diff2) & (~S0 | ~S3 | Diff0 | Diff1 | Diff2) & (
                                S1 | S3 | Diff1 | ~Diff3 | Diff4 | ~Diff5) & (
                                S1 | S3 | Diff2 | Diff3 | ~Diff4 | ~Diff5) & (
                                S1 | S3 | Diff0 | ~Diff3 | ~Diff4 | Diff5) & (
                                ~S1 | S3 | Diff0 | Diff3 | Diff4 | ~Diff5) & (
                                ~S1 | S3 | Diff1 | Diff3 | ~Diff4 | Diff5) & (
                                ~S1 | S3 | Diff2 | ~Diff3 | Diff4 | Diff5) & (
                                S1 | S3 | ~Diff0 | Diff1 | ~Diff2 | Diff4) & (
                                S1 | S3 | Diff0 | ~Diff1 | ~Diff2 | Diff5) & (
                                S1 | S3 | ~Diff0 | ~Diff1 | Diff2 | Diff3) & (
                                ~S1 | S3 | Diff0 | ~Diff1 | ~Diff2 | ~Diff5) & (
                                ~S1 | S3 | ~Diff0 | Diff1 | ~Diff2 | ~Diff4) & (
                                ~S1 | S3 | ~Diff0 | ~Diff1 | Diff2 | ~Diff3) & (
                                S1 | ~S2 | ~Diff0 | Diff1 | ~Diff2 | Diff4) & (
                                S1 | ~S2 | Diff0 | ~Diff1 | ~Diff2 | Diff5) & (
                                S1 | ~S2 | ~Diff0 | ~Diff1 | Diff2 | Diff3) & (
                                S1 | ~S3 | Diff0 | ~Diff1 | ~Diff2 | ~Diff5) & (
                                S1 | ~S3 | ~Diff0 | Diff1 | ~Diff2 | ~Diff4) & (
                                ~S2 | ~S3 | Diff1 | ~Diff3 | Diff4 | ~Diff5) & (
                                S1 | ~S3 | ~Diff0 | ~Diff1 | Diff2 | ~Diff3) & (
                                ~S2 | ~S3 | Diff2 | Diff3 | ~Diff4 | ~Diff5) & (
                                ~S2 | ~S3 | Diff0 | ~Diff3 | ~Diff4 | Diff5) & (
                                ~S1 | S2 | S3 | ~Diff3 | ~Diff4 | ~Diff5) & (
                                ~S1 | ~S2 | ~S3 | Diff3 | Diff4 | Diff5) & (
                                S1 | ~S2 | ~S3 | Diff0 | Diff1 | Diff2) & (
                                S0 | S2 | ~Diff0 | Diff3 | Diff4 | ~Diff5) & (
                                S0 | S2 | ~Diff1 | Diff3 | ~Diff4 | Diff5) & (
                                S0 | S2 | ~Diff2 | ~Diff3 | Diff4 | Diff5) & (
                                S0 | S1 | S2 | ~Diff3 | ~Diff4 | ~Diff5) & (
                                S0 | S1 | S2 | Diff3 | Diff4 | ~Diff5) & (
                                S0 | S1 | S2 | Diff3 | ~Diff4 | Diff5) & (
                                S0 | S1 | S2 | ~Diff3 | Diff4 | Diff5) & (
                                ~S1 | ~Diff0 | ~Diff1 | ~Diff2 | Diff3 | Diff4 | Diff5) & (
                                S1 | ~S2 | S3 | ~Diff0 | Diff3 | Diff4 | ~Diff5) & (
                                S1 | ~S2 | S3 | ~Diff1 | Diff3 | ~Diff4 | Diff5) & (
                                S1 | ~S2 | S3 | ~Diff2 | ~Diff3 | Diff4 | Diff5) & (
                                S1 | ~S2 | S3 | Diff0 | Diff1 | ~Diff2 | ~Diff3) & (
                                S1 | ~S2 | S3 | ~Diff0 | Diff1 | Diff2 | ~Diff5) & (
                                S1 | ~S2 | S3 | Diff0 | ~Diff1 | Diff2 | ~Diff4) & (
                                S1 | S3 | Diff0 | Diff1 | Diff2 | Diff3 | Diff4 | Diff5) & (
                                ~S1 | S3 | Diff0 | Diff1 | Diff2 | ~Diff3 | ~Diff4 | ~Diff5) & (
                                S1 | ~S2 | S3 | ~Diff0 | ~Diff1 | ~Diff2 | ~Diff3 | ~Diff4 | ~Diff5),
                        AllOne & NS0 == AllOne & (~Diff2 | Diff3) & (~Diff1 | Diff4) & (~Diff0 | Diff5) & (
                                Diff0 | Diff1 | Diff4 | ~Diff5) & (~Diff0 | Diff1 | Diff4 | ~Diff5) & (
                                Diff0 | ~Diff2 | ~Diff3 | Diff5) & (~Diff1 | Diff2 | Diff3 | ~Diff4) & (
                                Diff0 | Diff1 | Diff2 | Diff3) & (Diff0 | Diff1 | Diff2 | Diff5) & (
                                Diff0 | Diff1 | Diff2 | ~Diff3 | ~Diff4 | ~Diff5),
                        AllOne & NS1 == AllOne & (~Diff2 | Diff3) & (~Diff1 | Diff4) & (~Diff0 | Diff5) & (
                                Diff0 | ~Diff1 | ~Diff4 | ~Diff5) & (Diff1 | ~Diff2 | ~Diff3 | ~Diff4) & (
                                ~Diff0 | Diff2 | ~Diff3 | ~Diff5) & (
                                ~Diff0 | ~Diff1 | ~Diff2 | ~Diff3 | ~Diff4 | ~Diff5),
                        AllOne & NS2 == AllOne & (Diff3 | Diff4 | Diff5) & (Diff0 | Diff1 | Diff4 | ~Diff5) & (
                                Diff0 | Diff1 | Diff2 | Diff3) & (Diff0 | Diff1 | Diff2 | Diff5) & (
                                Diff0 | ~Diff1 | ~Diff4 | ~Diff5) & (Diff1 | ~Diff2 | ~Diff3 | ~Diff4) & (
                                ~Diff0 | Diff2 | ~Diff3 | ~Diff5) & (
                                ~Diff0 | ~Diff1 | ~Diff2 | ~Diff3 | ~Diff4 | ~Diff5),
                        AllOne & NS3 == AllOne & (Diff0 | ~Diff1 | ~Diff5) & (Diff1 | ~Diff2 | ~Diff4) & (
                                ~Diff0 | Diff2 | ~Diff3) & (~Diff0 | Diff1 | Diff4 | ~Diff5) & (
                                Diff0 | ~Diff2 | ~Diff3 | Diff5) & (~Diff1 | Diff2 | Diff3 | ~Diff4) & (
                                Diff0 | Diff1 | Diff2 | Diff3 | Diff4 | Diff5) & (
                                ~Diff0 | ~Diff1 | ~Diff2 | Diff3 | Diff4 | Diff5),
                        AllOne & HWPr == AllOne & (~Diff0 | ~Diff1 | ~Diff2 | ~Diff3 | ~Diff4 | ~Diff5) & (
                                Diff0 | Diff1 | Diff2 | ~Diff3 | ~Diff4 | ~Diff5) & (
                                Diff0 | Diff1 | Diff2 | Diff3 | Diff4 | Diff5) & (
                                ~Diff0 | ~Diff1 | ~Diff2 | Diff3 | Diff4 | Diff5),

                        ]

     
        Flag_Highest = (Flag >> (self._WordSize - 1)) & 1
        HWPr_Highest = (HWPr >> (self._WordSize - 1)) & 1
        S0_Highest = (S0 >> (self._WordSize - 1)) & 1
        S1_Highest = (S1 >> (self._WordSize - 1)) & 1
        S2_Highest = (S2 >> (self._WordSize - 1)) & 1
        S3_Highest = (S3 >> (self._WordSize - 1)) & 1
        v0Xorv2 = ((Diff0 ^ Diff1 ^ Diff2) >> (self._WordSize - 1)) & 1
        v0Xorv4 = ((Diff3 ^ Diff4 ^ Diff5) >> (self._WordSize - 1)) & 1

        Constraints += [
            Flag_Highest == (~S0_Highest | ~S2_Highest) & (~S0_Highest | v0Xorv4) & (
                    ~S0_Highest | ~S3_Highest | v0Xorv2) & (~S1_Highest | S2_Highest | ~v0Xorv2) & (
                    ~S1_Highest | ~S3_Highest | ~v0Xorv2) & (~S1_Highest | ~v0Xorv2 | v0Xorv4) & (
                    ~S1_Highest | S3_Highest | v0Xorv2 | ~v0Xorv4) & (
                    ~S2_Highest | ~S3_Highest | v0Xorv2 | v0Xorv4) & (
                    S1_Highest | ~S2_Highest | ~S3_Highest | v0Xorv2) & (
                    S1_Highest | S3_Highest | v0Xorv2 | v0Xorv4) & (
                    S0_Highest | S1_Highest | S2_Highest | ~v0Xorv4) & (
                    S1_Highest | ~S2_Highest | S3_Highest | ~v0Xorv2 | ~v0Xorv4),
            HWPr_Highest == 0,
        ]

        Constraints += [Flag == AllOne2, ]

        return Constraints

    def RoundCons_Boomerang_EBCT(self, StartRound):
   
        Constraints = []
        r0, r1, r2 = self._nrounds[0], self._nrounds[1], self._nrounds[2]
       
        assert (StartRound < (r0 + r1)) & (StartRound >= r0), "When searching for Boomerang, the number of intermediate rounds is set incorrectly!"

   
        InDiff = self._P[StartRound]
        OutDiff = self._P[StartRound + 1]
        HWPr = self._Pr[StartRound]

 

        Index = 0
        D0L, D0R, D3L, D3R = InDiff[Index + 1], InDiff[Index], InDiff[self._NumWord + Index + 1], InDiff[
            self._NumWord + Index],
        D1L, D1R, D2L, D2R = self.rotr(OutDiff[Index], self._WordSize, self._Alpha), OutDiff[Index + 3], \
                             self.rotr(OutDiff[self._NumWord + Index], self._WordSize, self._Alpha), OutDiff[
                                 self._NumWord + Index + 3],
        tempHWPr = HWPr[Index]
        Suffix = str(StartRound) + "_A" + str(Index)
        Constraints += self.Cons_Addition_EBCT(D0L, D0R, D3L, D3R, D1L, D1R, D2L, D2R, tempHWPr, Suffix)

        Index = 1
        D0L, D0R, D3L, D3R = InDiff[Index + 1], InDiff[Index], InDiff[self._NumWord + Index + 1], InDiff[
            self._NumWord + Index],
        D1L, D1R, D2L, D2R = self.rotl(OutDiff[Index], self._WordSize, self._Beta), InDiff[Index], \
                             self.rotl(OutDiff[self._NumWord + Index], self._WordSize, self._Beta), InDiff[
                                 self._NumWord + Index],
        tempHWPr = HWPr[Index]
        Suffix = str(StartRound) + "_A" + str(Index)
        Constraints += self.Cons_Addition_EBCT(D0L, D0R, D3L, D3R, D1L, D1R, D2L, D2R, tempHWPr, Suffix)

        Index = 2
        D0L, D0R, D3L, D3R = InDiff[Index + 1], InDiff[Index], InDiff[self._NumWord + Index + 1], InDiff[
            self._NumWord + Index],
        D1L, D1R, D2L, D2R = self.rotl(OutDiff[Index], self._WordSize, self._Gamma), InDiff[Index], \
                             self.rotl(OutDiff[self._NumWord + Index], self._WordSize, self._Gamma), InDiff[
                                 self._NumWord + Index],
        tempHWPr = HWPr[Index]
        Suffix = str(StartRound) + "_A" + str(Index)
        Constraints += self.Cons_Addition_EBCT(D0L, D0R, D3L, D3R, D1L, D1R, D2L, D2R, tempHWPr, Suffix)

        return Constraints

    def SearchBoomerangTrail_EBCT(self, HWPr=0, MaxHWPr=-1, ReverseSearchHWPr=False, AddMatSuiBound=False,
                                  MergeTrans=False, NoPr_BoomerangSwitch=False,
                                  ExcludeBoomerang=[], MaxPr_RK=-1, SetDiff=[[], [], []], E0ZeroOutDiff=False,
                                  SetDiff_E0=[]):
        Constraints = []
      
        if self.RelatedKey:  
            Constraints += self.DiffPropagationOfK(0, self._nrounds[0] - 1 + self._nrounds[1] - 1)

        Constraints += [(self._P[0][0] | self._P[0][1]) > 0]
        Constraints += [(self._P[-1][0] | self._P[-1][1]) > 0]
     
        Constraints += self.DiffPropagation(0, self._nrounds[0], RelatedKey=self.RelatedKey)

        

        for i in range(self._nrounds[0], self._nrounds[0] + self._nrounds[1]):
            Constraints += self.RoundCons_Boomerang_EBCT(StartRound=i, MergeTrans=MergeTrans)

     
        Constraints += self.DiffPropagation(self._nrounds[0] + self._nrounds[1], self._nrounds[2])

   
        r0 = self._nrounds[0]
        r1 = self._nrounds[1]
        r2 = self._nrounds[2]
   
        ObjHWDiff1 = 0
        for i in range(r0):
            ObjHWDiff1 += self.HW(self._Pr[i], self._WordSize)
          
            if AddMatSuiBound & (not RelatedKey):  
                if str(self._WordSize * 2) in self._BoundDiffPropagation.keys():
                    Bound = self._BoundDiffPropagation[str(self._WordSize * 2)]

                    Constraints += [ObjHWDiff1 >= (Bound[i + 1])]

 
        ObjBoomerangSwitch = 0
        for i in range(r0, r0 + r1):
            ObjBoomerangSwitch += self.HW(self._Pr[i], self._WordSize)
     
            if AddMatSuiBound:
                if str(self._WordSize * 2) in self._BoundDiffPropagation.keys():
                    Bound = self._BoundDiffPropagation[str(self._WordSize * 2)]
                    Constraints += [ObjBoomerangSwitch >= (Bound[i - r0 + 1])]

  
        ObjHWDiff2 = 0
        for i in range(r0 + r1, r0 + r1 + r2):
            ObjHWDiff2 += self.HW(self._Pr[i], self._WordSize)
        
            if AddMatSuiBound:
                if str(self._WordSize * 2) in self._BoundDiffPropagation.keys():
                    Bound = self._BoundDiffPropagation[str(self._WordSize * 2)]
                    Constraints += [ObjHWDiff2 >= (Bound[i - r0 - r1 + 1])]

   

        if RelatedKey:
            if (MaxPr_RK >= 0) & (r0 > 1):
                ObjHWRKDiff = 0
                for i in range(r0 - 1 + r1 - 1):
                    ObjHWRKDiff += self.HW(self._K_Pr[i], self._WordSize)
                Constraints += [MaxPr_RK >= ObjHWRKDiff]

    
        if NoPr_BoomerangSwitch:
            if ReverseSearchHWPr:
                Constraints += [(2 * ObjHWDiff1 + 2 * ObjHWDiff2) <= MaxHWPr]
            else:
                Constraints += [(2 * ObjHWDiff1 + 2 * ObjHWDiff2) == HWPr]
        else:
            if ReverseSearchHWPr:
                Constraints += [(2 * ObjHWDiff1 + ObjBoomerangSwitch + 2 * ObjHWDiff2) <= MaxHWPr]
            else:
                Constraints += [(2 * ObjHWDiff1 + ObjBoomerangSwitch + 2 * ObjHWDiff2) == HWPr]

  

        for i in range(len(ExcludeBoomerang)):
            Impossible = ExcludeBoomerang[i][-3]
            Counter = 0
            temp = 0
      
            for j in range(self._nrounds[0] + 1):
                for k in range(2):
                    temp = temp | (self._P[j][k] ^ Impossible[Counter])
                    Counter += 1
      

            for j in range(nrounds[1] - 1):
                for k in range(4):
                    temp = temp | (self._P[nrounds[0] + 1 + j][k] ^ Impossible[Counter])
                    Counter += 1

       
            for j in range(self._nrounds[0] + self._nrounds[1],
                           self._nrounds[0] + self._nrounds[1] + self._nrounds[2] + 1):
                for k in range(2):
                    temp = temp | (self._P[j][-2 + k] ^ Impossible[Counter])
                    Counter += 1

            if RelatedKey:  
     
                for i in range(nrounds[0] + m - 2 + max(0, nrounds[1] - 1)):
                    for j in range(len(self._K[i])):
                        temp = temp | (self._K[i][j] ^ Impossible[Counter])
                        Counter += 1
            Constraints += [temp >= 1]
      
        InDiffL = self._P[0][0]
        InDiffR = self._P[0][1]
        OutDiffL = self._P[-1][-2]
        OutDiffR = self._P[-1][-1]
        if len(SetDiff[0]) > 0:
            Constraints += [InDiffL ^ SetDiff[0][0] == 0,
                            InDiffR ^ SetDiff[0][1] == 0, ]
        if len(SetDiff[1]) > 0:
            Constraints += [OutDiffL ^ SetDiff[1][0] == 0,
                            OutDiffR ^ SetDiff[1][1] == 0, ]

   
        if len(SetDiff[2]) > 0:
           
            if len(SetDiff[2]) == 4:  
            
                Constraints += [self._K[2][0] ^ SetDiff[2][0] == 0,
                                self._K[1][0] ^ SetDiff[2][1] == 0,
                                self._K[0][0] ^ SetDiff[2][2] == 0,
                                self._K[0][1] ^ SetDiff[2][3] == 0,

                                ]
            else:

                Counter2 = 0
                for i in range(r0 + m - 2 + r1 - 1):
                    for j in range(len(self._K[i])):
                        Constraints += [self._K[i][j] ^ SetDiff[2][Counter2] == 0, ]
                        Counter2 += 1

   
        if E0ZeroOutDiff:
            Constraints += [self._P[r0][0] == 0,
                            self._P[r0][1] == 0,
                            ]

     
        if len(SetDiff_E0) > 0:
            for i in SetDiff_E0:
                Constraints += [self._P[i[0]][0] == i[1],
                                self._P[i[0]][1] == i[2],
                                ]

   
        for i in Constraints:
            self._Solver.add(i)

        Result = self._Solver.check()

        model = {}
        if Result:  
            model = self._Solver.model()
           
        return Result, model

    def SearchNB_EBCT(self, DiffPropa, HWPr=0, AddMatSuiBound=False, ExcludeNB=[], ExcludeNBMask=[], MaxHW_NB=-1,
                      MinHW_NB=-1, SearchRemaining=False, ExcludeTrails=[], NeutralDiff=[]):
        Constraints = []
   
        r0 = self._nrounds[0]
        r1 = self._nrounds[1]
        r2 = self._nrounds[2]

        assert (r0 == 0) & (r2 == 0), "When conducting a Neural Bit search, only nrounds[1] are considered valid here"
    
        Pr2 = [self.GetPr(i, suffix="_Diff") for i in range(r1)]
        Constraints += self.DiffPropagation(0, r1, Pr2, DiffPropa)

   
        for i in range(r1):
            Constraints += self.RoundCons_Boomerang_EBCT(StartRound=i)

        if len(NeutralDiff) > 0:
            for i in range(self._NumWord):
                Constraints += [self._P[0][self._NumWord + i] ^ NeutralDiff[i] == 0]

        ObjHWDiff = 0
        for i in range(r1):
            temp = []
            for pr in Pr2[i]:
                temp += self.HWList(pr, self._WordSize)
            ObjHWDiff += sum(temp)
  
            if AddMatSuiBound:
                Bound = self._BoundDiffPropagation["LEA128"]
                Constraints += [ObjHWDiff >= (Bound[i + 1])]


        ObjBoomerangSwitch = 0
        for i in range(r1):

            temp = []
            for pr in self._Pr[i]:
                temp += self.HWList(pr, self._WordSize)
            ObjBoomerangSwitch += sum(temp)

     
            if AddMatSuiBound:
                Bound = self._BoundDiffPropagation["LEA128"]
                Constraints += [ObjBoomerangSwitch >= (Bound[i + 1])]


        if SearchRemaining:
            Constraints += [(-ObjHWDiff + ObjBoomerangSwitch) >= HWPr]
        else:
            Constraints += [(-ObjHWDiff + ObjBoomerangSwitch) == HWPr]

   
        for i in range(len(ExcludeNB)):
            ND = self._P[0][self._NumWord:self._NumWord * 2]
            Impossible = ExcludeNB[i][0]
            Constraints += [((ND[0] ^ Impossible[0]) | (ND[1] ^ Impossible[1]) | (ND[2] ^ Impossible[2]) | (
                        ND[3] ^ Impossible[3])) >= 1]


        if ExcludeNBMask != []:
            for i in range(len(ExcludeNBMask)):


                ND = self._P[0][self._NumWord:self._NumWord * 2]
                MASK = ExcludeNBMask[i]
          
                if i == 0:
                    temp = ((MASK[0] & ND[0]) ^ (MASK[1] & ND[1]) ^ (MASK[2] & ND[2]) ^ (MASK[3] & ND[3]))
                    SUM = self.HW(temp, self._WordSize) & 1
                  
                else:
                    temp = ((MASK[0] & ND[0]) ^ (MASK[1] & ND[1]) ^ (MASK[2] & ND[2]) ^ (MASK[3] & ND[3]))
                    SUM = SUM | (self.HW(temp, self._WordSize) & 1)
                   

            Constraints += [SUM > 0]
     
        for Trail in ExcludeTrails:
            temp = 0
            Counter = 0
            for i in self._P:
                for j in i:
                    temp = temp | (j ^ Trail[Counter])
                    Counter += 1
            Constraints += [temp > 0]

  
        if MaxHW_NB > 0:
            print(" the maximum Hamming weight of a neutral bit:", MaxHW_NB)
            ND = self._P[0][self._NumWord:self._NumWord * 2]
            Constraints += [self.HW(ND[0], self._WordSize) + self.HW(ND[1], self._WordSize) + self.HW(ND[2],
                                                                                                      self._WordSize) + self.HW(
                ND[3], self._WordSize) <= MaxHW_NB]

        if MinHW_NB > 0:
            print(" the minimum Hamming weight of a neutral bit:", MinHW_NB)
            ND = self._P[0][self._NumWord:self._NumWord * 2]
            Constraints += [self.HW(ND[0], self._WordSize) + self.HW(ND[1], self._WordSize) + self.HW(ND[2],
                                                                                                      self._WordSize) + self.HW(
                ND[3], self._WordSize) >= MinHW_NB]

  
        for i in Constraints:
            self._Solver.add(i)


        Flag_UsingCryptominisat = self._Solver.useCryptominisat()


        Result = self._Solver.check()

        model = {}
        if Result: 
            model = self._Solver.model()
       
        return Result, model


def GetP(StartRound, R, nrounds, NumWord=4):
    r0 = nrounds[0]
    r1 = nrounds[1]
    r2 = nrounds[2]
    if (R < r0) | (R > (r0 + r1)):
        P = ["p" + str(StartRound + R) + "_" + str(i) for i in range(NumWord)]
    elif r1 == 0:
        P = ["p" + str(StartRound + R) + "_" + str(i) for i in range(NumWord)]
    else:
        P = ["p" + str(StartRound + R) + "_" + str(i) for i in range(NumWord * 2)]
    return P


def GetPr(StartRound, R, Suffix="", VarName="pr", NumAddition=3):
    Pr = [VarName + str(StartRound + R) + "_" + str(i) + Suffix for i in range(NumAddition)]
    return Pr


def GetK(R, nrounds, VarName="k"):


    r0 = nrounds[0]
    r1 = nrounds[1]
    if (R < (r0 + max(0, r1 - 1))):
        K = [VarName + str(R) + "L", VarName + str(R) + "R"]
    else:
        K = [VarName + str(R) + "L", ]

    return K


def hw(X, WordSize=16):
    Counter = 0
    for i in range(WordSize):
        if (X >> i) & 1 == 1:
            Counter += 1
    return Counter


def DisplayDiffPropagation(model, nrounds, ExpectedHWPr,
                           m=4, WordSize=32, FileName=None, RelatedKey=False,
                           NoPr_BoomerangSwitch=False):
    StartRound = 0
    R = nrounds[0] + nrounds[1] + nrounds[2]
    P = [GetP(StartRound, i, nrounds) for i in range(R + 1)]
    Pr = [GetPr(StartRound, i) for i in range(R)]

 
    HWPr = 0
    for i in range(R):
        for j in range(len(Pr[i])):
            HWPr += hw(model[Pr[i][j]], WordSize)

    print(str(R) + "-round cipher, the preliminary estimate of the differential weight:\t", HWPr)
    print(str(R) + "-round cipher, the expected value of differential weight:\t", ExpectedHWPr)

    print("The starting round is:\t", str(StartRound))

    print("**********************")
    print("Differential propagation")
    for i in range(R + 1):
        for j in range(len(P[i])):
            temp = "{:0" + str(WordSize) + "b}"
            print(temp.format(model[P[i][j]]), end="\t")
        print()
    print("**********************")
    print("Corresponding probability")
    for i in range(R):
        for j in range(len(Pr[i])):
            temp = "{:0" + str(WordSize) + "b}"
            print(temp.format(model[Pr[i][j]]), end="\t")
        print()


    if FileName != None:
        WritingFile = open(FileName, "a")
        print(str(R) + "-round cipher, the preliminary estimate of the differential weight::\t", HWPr, file=WritingFile)
        print("The starting round is: \t", str(StartRound), file=WritingFile)
        P = [GetP(StartRound, i, nrounds) for i in range(R + 1)]
        Pr = [GetPr(StartRound, i) for i in range(R)]
        print("**********************", file=WritingFile)
        print("Differential propagation", file=WritingFile)
        for i in range(R + 1):
            for j in range(len(P[i])):
                temp = "{:0" + str(WordSize) + "b}"
                print(temp.format(model[P[i][j]]), end="\t", file=WritingFile)

            print("", file=WritingFile)

        print("**********************", file=WritingFile)
        print("Corresponding probability", file=WritingFile)
        for i in range(R):
            for j in range(len(Pr[i])):
                temp = "{:0" + str(WordSize) + "b}"
                print(temp.format(model[Pr[i][j]]), end="\t", file=WritingFile)
            print("", file=WritingFile)

     
        temp = "{:0" + str(WordSize) + "b}"
        for i in range(nrounds[1]):
            print("Em's", i, "-round", file=WritingFile)
            for j in range(8):
                StateName = "S" + str(j) + "_R" + str(i + nrounds[0])
                print(StateName, temp.format(model[StateName]), file=WritingFile)
            FlagName = "Flag_R" + str(i + nrounds[0])
            print(FlagName, temp.format(model[FlagName]), file=WritingFile)

        WritingFile.close()
    temp = "0x{:0" + str(int(WordSize / 4)) + "x}"

    return [model[P[0][i]] for i in range(4)] + [model[P[-1][i]] for i in range(4)] + [HWPr]


def bit_inv(a, n):
    a = a.astype(np.uint8)
    inv = np.eye(n, dtype=np.uint8)


    for i in range(n):
        if a[i, i] == 0:
            for j in range(i + 1, n):
                if a[j, i] == 1:
                    a[[i, j], :] = a[[j, i], :]
                    inv[[i, j], :] = inv[[j, i], :]
                    break
            else:
                print(i, j)
                print(a)
                print("The rank of the current matrix is")
                print(np.linalg.matrix_rank(a))
                assert 1 < 0, ("Matrix irreversibility")

        for j in range(i + 1, n):
            if a[j, i] == 1:
                a[j] ^= a[i]
                inv[j] ^= inv[i]

    for i in range(n - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            if a[j, i] == 1:
                a[j] ^= a[i]
                inv[j] ^= inv[i]

    return inv


def bit_rank(a, n, m):
    a = a.astype(np.uint8)
    rank = 0
    Index = 0

    for i in range(n):
        if rank == m:
            break

        for j in range(Index, m):
            row = -1
            for k in range(rank, n):
                if a[k, j] == 1:
                    row = k
                    Index = j + 1

                    a[[row, rank], :] = a[[rank, row], :]
                    break
            if row >= 0:
                break
        else:
            return rank
        rank += 1
        for j in range(rank, n):
            if a[j, Index - 1] == 1:
                a[j] ^= a[rank - 1]

    return rank


def GenBaseVector2(ObtainedVector, n):
    BaseVector = ObtainedVector.copy()

    # print(BaseVector)
    assert len(BaseVector) == bit_rank(BaseVector, len(BaseVector), n), "The rows of the input matrix are not linearly independent!" + str(
        len(BaseVector)) + "______" + str(bit_rank(BaseVector, len(BaseVector), n))
    for i in range(n):
        OneVector = np.zeros((1, n), dtype=np.int_)
        OneVector[0, n - i - 1] = 1
        # print(BaseVector)
        # print(OneVector)
        temp = np.concatenate((BaseVector, OneVector))
        if bit_rank(temp, len(temp), n) > bit_rank(BaseVector, len(BaseVector), n):
            BaseVector = temp.copy()
            if bit_rank(temp, len(temp), n) == n:
                BaseVector = BaseVector.T

                Inv_BaseVector = bit_inv(BaseVector, n) % 2
                Inv_BaseVector = np.array(Inv_BaseVector, dtype=np.int_)

                return BaseVector, Inv_BaseVector

    while (bit_rank(BaseVector, len(BaseVector), n) < n):
        OneVector = np.random.randint(2, size=(1, n))
        temp = np.concatenate((BaseVector, OneVector))
        if bit_rank(temp, len(temp), n) > bit_rank(BaseVector, len(BaseVector), n):
            BaseVector = temp.copy()

    BaseVector = BaseVector.T

    Inv_BaseVector = bit_inv(BaseVector, n) % 2
    Inv_BaseVector = np.array(Inv_BaseVector)
    print("!!!!!!!!!!!2 !!!!!!!!!!!")
    print(np.matmul(BaseVector, Inv_BaseVector) % 2)
    return BaseVector, Inv_BaseVector


def Mask2Bin(MASKs, WordSize=32):
    NumWord = len(MASKs)
    n = NumWord * WordSize

    Vector = np.zeros((1, n), dtype=np.int_)
    for i in range(NumWord):
        for j in range(WordSize):
            Vector[0, i * WordSize + j] = int(MASKs[i] >> int(WordSize - 1 - j)) & 1
    return Vector


def Bin2Mask(BinNum, WordSize=32, NumWord=4):
 
    Mask = [0 for i in range(NumWord)]

    for i in range(NumWord):
        for j in range(WordSize):
            Mask[i] += BinNum[i * WordSize + j] << (WordSize - 1 - j)

    return Mask


def ExcludeNB_Mask(NB_Obtained, WordSize=32, NumWord=4):
   

    ObtainedVectors = Mask2Bin(NB_Obtained[0][0])

    for NB in NB_Obtained[1:]:
        temp = Mask2Bin(NB[0])
        ObtainedVectors = np.concatenate((ObtainedVectors, temp))

    BaseVectors, Inv_BaseVectors = GenBaseVector2(ObtainedVectors, WordSize * NumWord)


    ExcludeMask = []
    for i in range(len(NB_Obtained), WordSize * NumWord):
        Mask = Bin2Mask(Inv_BaseVectors[i, :])
        ExcludeMask.append(Mask)

    return ExcludeMask


def DisplayDiffPropagation_NB(model, HWPr, nrounds, WordSize=32, NumWord=4, ReturnTrail=False, FileName=None):
    StartRound = 0
    R = nrounds[0] + nrounds[1] + nrounds[2]
    print(str(R) + "-round cipher, the differential weight is:\t", HWPr)
    print("The starting round：\t", str(StartRound))
    P = [GetP(StartRound, i, nrounds) for i in range(R + 1)]
    Pr = [GetPr(StartRound, i, ) for i in range(R)]

    Pr2 = [GetPr(StartRound, i, Suffix="_Diff") for i in range(R)]
    print("**********************")
    print("Differential propagation")
    for i in range(R + 1):
        for j in range(len(P[i])):
            temp = "{:0" + str(WordSize) + "b}"
            print(temp.format(model[P[i][j]]), end="\t")

        print()

    print("**********************")
    print("The corresponding probability in the Boomerang Switch")
    for i in range(R):
        for j in range(len(Pr[i])):
            temp = "{:0" + str(WordSize) + "b}"
            print(temp.format(model[Pr[i][j]]), end="\t")
        print()

    print("**********************")
    print("Probability corresponding to differential propagation")
    for i in range(R):
        for j in range(len(Pr2[i])):
            temp = "{:0" + str(WordSize) + "b}"
            print(temp.format(model[Pr2[i][j]]), end="\t")
        print()

    if FileName != None:
        WritingFile = open(FileName, "a")
        print(str(R) + "轮的差分重量是:\t", HWPr, file=WritingFile)
        print("开始轮是：\t", str(StartRound), file=WritingFile)
        P = [GetP(StartRound, i, nrounds) for i in range(R + 1)]
        Pr = [GetPr(StartRound, i, ) for i in range(R)]

        Pr2 = [GetPr(StartRound, i, Suffix="_Diff") for i in range(R)]
        print("**********************", file=WritingFile)
        print("差分传播", file=WritingFile)
        for i in range(R + 1):
            for j in range(len(P[i])):
                temp = "{:0" + str(WordSize) + "b}"
                print(temp.format(model[P[i][j]]), end="\t", file=WritingFile)

            print("", file=WritingFile)

        print("**********************", file=WritingFile)
        print("Boomerang Switch中对应的概率", file=WritingFile)
        for i in range(R):
            for j in range(len(Pr[i])):
                temp = "{:0" + str(WordSize) + "b}"
                print(temp.format(model[Pr[i][j]]), end="\t", file=WritingFile)
            print("", file=WritingFile)

        print("**********************", file=WritingFile)
        print("差分传播中对应的概率", file=WritingFile)
        for i in range(R):
            for j in range(len(Pr2[i])):
                temp = "{:0" + str(WordSize) + "b}"
                print(temp.format(model[Pr2[i][j]]), end="\t", file=WritingFile)
            print("", file=WritingFile)

        WritingFile.close()

    NeutralDiffSearched = [model[P[0][NumWord + i]] for i in range(NumWord)]
    if ReturnTrail:
        #
        Trail = []
        for i in P:
            for j in i:
                Trail.append(model[j])
        Trail.append(HWPr)

        return NeutralDiffSearched, Trail
    else:
        return NeutralDiffSearched


def SearchGoodNB(nrounds, DiffPropa, WordSize=32, AddMatSuiBound=False, ExcludeNB=[], ExcludeNBMask=[], MinHWPr=-1,
                 MaxHW_NB=-1, MinHW_NB=-1, UpperBoundHWPr=100, WritingToFile=None):
    HWPr = max(0, MinHWPr)
    Flag = True

    StartTime = time.time()
    SearchRemaining = False
    while (Flag):
        print("LEA", WordSize * 2, "'s structure is:\t", nrounds)
        print("The target value for probability weight is", HWPr)
        DiffModel = LEA(nrounds=nrounds, WordSize=WordSize)
        Result, model = DiffModel.SearchNB_EBCT(DiffPropa, HWPr, AddMatSuiBound, ExcludeNB, ExcludeNBMask, MaxHW_NB,
                                                MinHW_NB, SearchRemaining=SearchRemaining)
        print(Result)
        if Result:
            print("All variables have the following values")
            print(model)
            ValidNB = DisplayDiffPropagation_NB(model=model, HWPr=HWPr, nrounds=nrounds, WordSize=WordSize,
                                                FileName=WritingToFile)
            Flag = False
        HWPr += 1
        EndTime = time.time()
        print("The time used is", EndTime - StartTime, "s")

 
        if HWPr > UpperBoundHWPr:
            if Flag == False:  
                print("When HWPr>=", UpperBoundHWPr, "There is still a solution, but we will no longer search!")
                return False, ValidNB
            elif HWPr == (UpperBoundHWPr + 1):
                SearchRemaining = True
            else:
                print("******************")
                print("There is no solution in the remaining space!")
                print("******************")
                return False, []


    return True, [ValidNB, HWPr - 1]


def AddExcludePoint(ExcludeNB, ValidNB):
    Flag = True
    for i in ExcludeNB:
        if (i[0] == ValidNB[0]) & (i[1] == ValidNB[1]):
            Flag = False
            break
    tempValidNB = ValidNB.copy()
    hw_NB = hw(ValidNB[0]) + hw(ValidNB[1])

    if not Flag:
        return False, ExcludeNB, ExcludeNB
    else:
        
        NewExcludeNB = ExcludeNB.copy()
        NewExcludeNB.append(ValidNB)
        for i in ExcludeNB:
            NewExcludeNB.append([i[0] ^ ValidNB[0], i[1] ^ ValidNB[1], 0, 0])

       
            temp_hw_NB = hw(i[0] ^ ValidNB[0]) + hw(i[1] ^ ValidNB[1])
            if temp_hw_NB < hw_NB:
                hw_NB = temp_hw_NB
                tempValidNB[0], tempValidNB[1] = i[0] ^ ValidNB[0], i[1] ^ ValidNB[1]

        return True, NewExcludeNB, tempValidNB


def modify_matrix(matrix):
    rows, cols = matrix.shape

    Num = 0
    for j in range(cols):
        i = Num
        while ((i < rows) and (matrix[i, j] == 0)):  
     
            i += 1
        if i == rows:
            continue



        if i != Num:
            temp = matrix[Num].copy()
            matrix[Num] = matrix[i]
            matrix[i] = temp


        for y in range(rows):
            if y == Num:
                continue
            if matrix[y, j] == 1:
                matrix[y] ^= matrix[Num]
        Num += 1

    return matrix





if __name__ == "__main__":
    SearchDiff = False
    SearchNeuralBitIterative = True



    GenBasis = False

    if SearchDiff:
    
        SearchNum = 2 ** 5
     
        SpecifyDiff = [[], [0x0, 0x0, 0x0, 0x8000_0000]]
        SpecifyDiff = [[], [0x8000_0000, 0x8000_0000, 0x8000_0000, 0x8000_0000]]
 

        DiffPropaList = []
        nrounds = [3, 0, 0]
        HWPr = 33
        WordSize = 32


        HWInDiff = -1
        HWOutDiff = -1

        AddMatSuiBound = True
        while (len(DiffPropaList) < SearchNum):

            StartTime = time.time()
            DiffModel = LEA(nrounds=nrounds)
            Result, model = DiffModel.SearchDiff(HWPr, AddMatSuiBound, ExcludePoint=DiffPropaList,
                                                 SpecifyDiff=SpecifyDiff, HWInDiff=HWInDiff, HWOutDiff=HWOutDiff)
            print(HWPr, Result)
            if Result:
                print("All variables have the following values")
                print(model)
                TEMP = DisplayDiffPropagation(model=model, ExpectedHWPr=HWPr, nrounds=nrounds, WordSize=WordSize)
                temp = "{:0" + str(int(WordSize / 4)) + "x}"
                DiffPropaList.append(TEMP)
                EndTime = time.time()
                print("The time used is", EndTime - StartTime, "s")
                StartTime = time.time()
            else:
                HWPr += 1

        print("The trails searched is ", SearchNum,)
        tempDiffPropaList = [["0x" + temp.format(i[j]) if j < 8 else i[j] for j in range(9)] for i in DiffPropaList]
        print(tempDiffPropaList)

    if SearchNeuralBitIterative:
  

        Method_ExcludeNB = 1 

        WordSize = 32
        NumWord = 4
        SearchNum = WordSize * NumWord

        WritingToFile = None  
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y%m%d%H%M%S")
        ResultPath = "./Result/"
        WritingToFile = ResultPath + formatted_time + ".txt"  

 
        nrounds = [0, 3, 0]
        DiffPropa = [
            [0x0a000080, 0x00402080, 0x00402010, 0x40402011],
            [0x80400014, 0x8000000c, 0x78000004, 0x0a000080],
            [0x80000000, 0x80400000, 0x80400010, 0x80400014],
            [0x80000000, 0x80000000, 0x80000000, 0x80000000],
        ]

        nrounds = [0, 3, 0]
        DiffPropa = [
            [0x0a000080, 0x00002080, 0x00002000, 0x40002020],
            [0x7fc00014, 0x0000000c, 0x08000004, 0x0a000080],
            [0x80000000, 0x80400000, 0x80400010, 0x7fc00014],
            [0x80000000, 0x80000000, 0x80000000, 0x80000000],

        ]



        MaxHW_NB = -1
        MinHW_NB = -1

        AddMatSuiBound = True
        NB_Obtained = [

        ]

        if len(DiffPropa) != 0:
            NB_Obtained += [[DiffPropa[0], 0, 0]]  

        ExcludeNB = []  
        ExcludeNB += NB_Obtained

        while (len(NB_Obtained) < SearchNum):
            StartAllTime = time.time()
            MinHWPr = -1
            if len(NB_Obtained) > 1:
                MinHWPr = NB_Obtained[-1][-2]
            else:
                MinHWPr = 0
            ExcludeNBMask = []
            if Method_ExcludeNB != 0:
                ExcludeNBMask = ExcludeNB_Mask(NB_Obtained, WordSize)
                print("ExcludeNBMask:")
                TEMP = "{:0" + str(int(WordSize / 4)) + "x}"
                temp = [[TEMP.format(j) for j in i] for i in ExcludeNBMask]
                print(temp)
                ExcludeNB_Num = 1
            else:
                ExcludeNB_Num = len(ExcludeNB)
            FlagContinue, ValidNB = SearchGoodNB(nrounds, WordSize=WordSize, AddMatSuiBound=AddMatSuiBound,
                                                 DiffPropa=DiffPropa, ExcludeNB=ExcludeNB[:ExcludeNB_Num],
                                                 ExcludeNBMask=ExcludeNBMask, MinHWPr=MinHWPr,
                                                 MaxHW_NB=MaxHW_NB, MinHW_NB=MinHW_NB, WritingToFile=WritingToFile)

            EndAllTime = time.time()
            print("花费总时间为：", EndAllTime - StartAllTime, "s")
            ValidNB.append(EndAllTime - StartAllTime)

            if Method_ExcludeNB != 0:
                if FlagContinue:
                    NB_Obtained.append(ValidNB)
            else:
                Flag, ExcludeNB, ValidNB = AddExcludePoint(ExcludeNB, ValidNB)
                if Flag:
                    NB_Obtained.append(ValidNB)

    
            temp = "{:0" + str(int(WordSize / 4)) + "x}"
            print("NB_Obtained:", len(NB_Obtained))
            print("NB_Obtained:", NB_Obtained)
            temp1 = [[i[j] if j >= 1 else ["0x" + temp.format(k) for k in i[0]] for j in range(len(i))] for i in
                     NB_Obtained[:]]
            print(temp1)
            if not FlagContinue:
                print("There is no way to search for results again!")
                break

        print("***************************")
        print("All search results are as follows")

        temp = "{:0" + str(int(WordSize / 4)) + "x}"
        print("ExcludeNB length:", len(ExcludeNB))

        print("NB_Obtained length:", len(NB_Obtained),)
        NB_Obtained = [[i[j] if j >= 1 else ["0x" + temp.format(k) for k in i[0]] for j in range(len(i))] for i in
                       NB_Obtained[:]]
        print(NB_Obtained)

       
        if WritingToFile != None:
            WritingFile = open(WritingToFile, "a")
            print("NB_Obtained length:", len(NB_Obtained), file=WritingFile)
            print(NB_Obtained, file=WritingFile)
            WritingFile.close()




 
