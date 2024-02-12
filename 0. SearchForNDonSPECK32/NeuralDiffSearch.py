import stp
import time
import numpy as np
import math as mt
import BCT_UBCT_LBCT_EBCT as Matrice

np.set_printoptions(threshold=np.inf)



class SPECK():
    def __init__(self, nrounds, m=4, WordSize=16, BoundDiffPropagation={}, RelatedKey=False):

        self._nrounds = nrounds  
        self._m = m  
        self._WordSize = WordSize
       
        self.RelatedKey = RelatedKey

        if WordSize == 16:
            self._Alpha, self._Beta = 7, 2
        else:
            self._Alpha, self._Beta = 8, 3
      
        self._Solver = stp.Solver()
 
        self._P = [self.GetP(i) for i in range(nrounds[0] + nrounds[1] + nrounds[2] + 1)]

        self._Pr = [self.GetPr(i) for i in range(nrounds[0] + nrounds[1] + nrounds[2])]

 
        self._K = [self.GetK(i) for i in
                   range(nrounds[0] + self._m - 2 + max(0, nrounds[1] - 1))]  
        self._K_Pr = [self.GetPr(i, VarName="k_pr") for i in range(nrounds[0] - 1 + max(0, nrounds[1] - 1))]


        if (len(BoundDiffPropagation) == 0):
            self._BoundDiffPropagation = {
                "32": [0, 0, 1, 3, 5, 9, 13, 18, 24, 30, 34, 38, 42, 45, 49, 54, 58, 63, 69, 74, 77, 81, 85],
                "48": [0, 0, 1, 3, 6, 10, 14, 19, 26, 33, 40, 45, 49, 54, 58, 63, 68, 75, 82],
                "64": [0, 0, 1, 3, 6, 10, 15, 21, 29, 34, 38, 42, 46, 50, 56, 62, 70, 73, 76, 81, 85, 89, 94, 99, 107,
                       112, 116, 121],
                "96": [0, 0, 1, 3, 6, 10, 15, 21, 30, 39, 49],
                "128": [0, 0, 1, 3, 6, 10, 15, 21, 30, 39],
            }
        else:
            self._BoundDiffPropagation = BoundDiffPropagation

    def GetP(self, R, ):

        r0 = self._nrounds[0]
        r1 = self._nrounds[1]
        r2 = self._nrounds[2]
        if (R < r0) | (R > (r0 + r1)):
            P = [self._Solver.bitvec("p" + str(R) + "L", self._WordSize),
                 self._Solver.bitvec("p" + str(R) + "R", self._WordSize), ]
        elif r1 == 0:
            P = [self._Solver.bitvec("p" + str(R) + "L", self._WordSize),
                 self._Solver.bitvec("p" + str(R) + "R", self._WordSize), ]
        else:

            P = [self._Solver.bitvec("p" + str(R) + "_0", self._WordSize),
                 self._Solver.bitvec("p" + str(R) + "_1", self._WordSize),
                 self._Solver.bitvec("p" + str(R) + "_2", self._WordSize),
                 self._Solver.bitvec("p" + str(R) + "_3", self._WordSize)
                 ]
        return P

    def GetPr(self, R, VarName="pr", suffix=""):
  
        Pr = self._Solver.bitvec(VarName + str(R) + suffix, self._WordSize)
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

    def RoundCons_DiffPropagation(self, R, Pr2=[], NBSearch=False, RelatedKey=False):
        Constraints = []
        r0, r1, r2 = self._nrounds[0], self._nrounds[1], self._nrounds[2]
        if (r1 != 0) & (R == (r0 + r1)) & (~NBSearch):
            X = self._P[R][2:4]

        else:
            X = self._P[R][:2]

        Y = self._P[R + 1][:2] 
        if len(Pr2) != 0:
            Pr = Pr2[R]
        else:
            Pr = self._Pr[R]

        n = self._WordSize
        x0 = X[0]
        x1 = X[1]


        if RelatedKey:
            y0 = Y[0] ^ self._K[R][1]
            y1 = Y[1] ^ self._K[R][1]
        else:
            y0 = Y[0]
            y1 = Y[1]

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
            Constraints += self.RoundCons_DiffPropagation(i, Pr, NBSearch=NBSearch, RelatedKey=RelatedKey)
       
        if len(DiffPropa) != 0:
            for i in range(len(DiffPropa)):
                
                if DiffPropa[i] < 0:
                    continue

                if i < 2:
                    Constraints += [(self._P[StartRound][i] ^ DiffPropa[i]) == 0]
                else:
                    Constraints += [(self._P[StartRound + R][i - 2] ^ DiffPropa[i]) == 0]

        return Constraints

    def SearchDiff(self, HWPr=0, AddMatSuiBound=False, ExcludePoint=[], SpecifyDiff=[[], []]):

     
        Constraints = self.DiffPropagation(0, self._nrounds[0])
     
        ObjList = []

        for i in range(len(self._Pr)):
            if i == 0:
                ObjList.append(self.HW(self._Pr[i], self._WordSize))
            else:
                ObjList.append(self.HW(self._Pr[i], self._WordSize) + ObjList[-1])

        Constraints += [ObjList[-1] == HWPr]

   
        if AddMatSuiBound:
            if str(self._WordSize * 2) in self._BoundDiffPropagation.keys():
                Bound = self._BoundDiffPropagation[str(self._WordSize * 2)]
                for i in range(len(self._Pr) - 1):
                    Constraints += [ObjList[i] >= Bound[i + 1]]

  
        for i in range(len(ExcludePoint)):
            InDiffL = self._P[0][0]
            InDiffR = self._P[0][1]
            OutDiffL = self._P[-1][-2]
            OutDiffR = self._P[-1][-1]

            Impossible = ExcludePoint[i]
            Constraints += [((InDiffL ^ Impossible[0]) | (InDiffR ^ Impossible[1]) | (OutDiffL ^ Impossible[2]) | (
                    OutDiffR ^ Impossible[3])) >= 1]

   
        if len(SpecifyDiff[0]) > 0:
            InDiffL = self._P[0][0]
            InDiffR = self._P[0][1]
            Constraints += [((InDiffL ^ SpecifyDiff[0][0]) | (InDiffR ^ SpecifyDiff[0][1])) == 0]

        if len(SpecifyDiff[1]) > 0:
            OutDiffL = self._P[-1][0]
            OutDiffR = self._P[-1][1]
            Constraints += [((OutDiffL ^ SpecifyDiff[1][0]) | (OutDiffR ^ SpecifyDiff[1][1])) == 0]

   
        for i in Constraints:
            self._Solver.add(i)

        Result = self._Solver.check()

        model = {}
        if Result:  
            model = self._Solver.model()
          

        return Result, model

    def RoundCons_Boomerang_EBCT(self, StartRound, MergeTrans=False):
  
        Constraints = []
        r0, r1, r2 = self._nrounds[0], self._nrounds[1], self._nrounds[2]
 
        assert (StartRound < (r0 + r1)) & (StartRound >= r0), "When searching for Boomerang, the number of intermediate rounds is set incorrectly!"


        InDiff = self._P[StartRound]
        OutDiff = self._P[StartRound + 1]

        D0L, D0R, D3L, D3R = self.rotr(InDiff[0], self._WordSize, self._Alpha), InDiff[1], \
                             self.rotr(InDiff[2], self._WordSize, self._Alpha), InDiff[3]

        D1L, D1R, D2L, D2R = OutDiff[0], self.rotr(OutDiff[0] ^ OutDiff[1], self._WordSize, self._Beta), \
                             OutDiff[2], self.rotr(OutDiff[2] ^ OutDiff[3], self._WordSize, self._Beta)

        HWPr = self._Pr[StartRound]

        Constraints += [D0R == D1R, D3R == D2R, ]
        
        S0 = self._Solver.bitvec("S0_R" + str(StartRound), self._WordSize)
        S1 = self._Solver.bitvec("S1_R" + str(StartRound), self._WordSize)
        S2 = self._Solver.bitvec("S2_R" + str(StartRound), self._WordSize)
        S3 = self._Solver.bitvec("S3_R" + str(StartRound), self._WordSize)
        Flag = self._Solver.bitvec("Flag_R" + str(StartRound), self._WordSize)

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
        if MergeTrans:
            Constraints += [
                AllOne & Flag == AllOne & (~S0 | ~S1) & (~S0 | ~S2) & (~S1 | Diff0 | Diff1 | ~Diff2) & (
                        ~S0 | ~Diff3 | Diff4 | ~Diff5) & (~S0 | ~Diff3 | ~Diff4 | Diff5) & (
                        ~S0 | Diff3 | Diff4 | Diff5) & (~S0 | Diff3 | ~Diff4 | ~Diff5) & (
                        ~S0 | S3 | ~Diff2 | ~Diff3) & (Diff0 | ~Diff1 | Diff2 | ~Diff3 | Diff5) & (
                        Diff0 | ~Diff2 | Diff3 | ~Diff4 | Diff5) & (~Diff0 | Diff1 | Diff2 | ~Diff3 | Diff4) & (
                        Diff0 | ~Diff1 | Diff3 | ~Diff4 | ~Diff5) & (Diff1 | ~Diff2 | Diff3 | Diff4 | ~Diff5) & (
                        ~Diff0 | Diff1 | Diff3 | ~Diff4 | ~Diff5) & (~Diff1 | Diff2 | Diff3 | Diff4 | ~Diff5) & (
                        ~Diff0 | Diff2 | Diff3 | ~Diff4 | Diff5) & (S3 | Diff1 | ~Diff2 | ~Diff3 | ~Diff4) & (
                        ~S1 | ~Diff1 | ~Diff3 | Diff4 | ~Diff5) & (~S1 | ~Diff0 | ~Diff3 | ~Diff4 | Diff5) & (
                        ~S1 | Diff0 | ~Diff1 | Diff2 | Diff4) & (~S1 | ~Diff0 | Diff1 | Diff2 | Diff5) & (
                        ~S2 | Diff0 | ~Diff1 | ~Diff3 | Diff5) & (~S1 | S3 | Diff0 | ~Diff2 | ~Diff5) & (
                        ~S1 | ~Diff0 | ~Diff1 | ~Diff2 | Diff3) & (~S2 | ~Diff0 | ~Diff1 | Diff4 | ~Diff5) & (
                        ~S2 | S3 | Diff0 | Diff1 | ~Diff2) & (~S1 | ~S3 | Diff0 | ~Diff1 | Diff2) & (
                        ~S1 | ~S3 | ~Diff0 | ~Diff1 | ~Diff2) & (~S0 | ~S3 | Diff0 | Diff1 | Diff2) & (
                        ~S1 | S2 | ~S3 | Diff0 | ~Diff5) & (S0 | S2 | Diff3 | Diff4 | ~Diff5) & (
                        S1 | Diff0 | ~Diff1 | ~Diff2 | Diff4 | Diff5) & (
                        S1 | S3 | Diff0 | ~Diff3 | ~Diff4 | Diff5) & (
                        S1 | ~Diff0 | Diff1 | ~Diff2 | Diff4 | Diff5) & (
                        S1 | S3 | Diff1 | ~Diff3 | Diff4 | ~Diff5) & (~S1 | S3 | Diff0 | Diff3 | Diff4 | ~Diff5) & (
                        ~S1 | S3 | Diff2 | ~Diff3 | Diff4 | Diff5) & (~S1 | S3 | Diff1 | Diff3 | ~Diff4 | Diff5) & (
                        ~S1 | S3 | ~Diff0 | ~Diff1 | Diff2 | ~Diff3) & (
                        S1 | S3 | Diff2 | Diff3 | ~Diff4 | ~Diff5) & (
                        ~S2 | ~S3 | Diff0 | ~Diff3 | ~Diff4 | Diff5) & (
                        ~S2 | ~S3 | Diff1 | ~Diff3 | Diff4 | ~Diff5) & (
                        ~S1 | ~S2 | ~Diff0 | Diff1 | Diff4 | Diff5) & (
                        ~S1 | S2 | ~Diff2 | ~Diff3 | Diff4 | Diff5) & (
                        S0 | S3 | Diff0 | ~Diff2 | Diff4 | ~Diff5) & (
                        ~S2 | S3 | ~Diff0 | ~Diff1 | ~Diff4 | Diff5) & (
                        S1 | ~S3 | ~Diff0 | ~Diff1 | Diff2 | ~Diff3) & (
                        S1 | ~S3 | ~Diff0 | Diff1 | ~Diff2 | ~Diff4) & (
                        S2 | ~S3 | Diff0 | ~Diff1 | ~Diff2 | ~Diff5) & (
                        ~S2 | ~S3 | Diff2 | Diff3 | ~Diff4 | ~Diff5) & (
                        ~S1 | S2 | S3 | ~Diff3 | ~Diff4 | ~Diff5) & (S1 | ~S2 | ~Diff0 | ~Diff1 | Diff2 | Diff3) & (
                        S0 | S2 | Diff0 | Diff1 | ~Diff2 | ~Diff3) & (~S1 | S2 | S3 | Diff3 | ~Diff4 | Diff5) & (
                        S1 | ~S2 | ~S3 | Diff0 | Diff1 | Diff2) & (S1 | ~S2 | S3 | ~Diff0 | ~Diff2 | ~Diff3) & (
                        S0 | ~S3 | ~Diff0 | Diff1 | Diff2 | ~Diff5) & (~S1 | ~S2 | ~S3 | Diff3 | Diff4 | Diff5) & (
                        S0 | S1 | S3 | ~Diff0 | Diff1 | ~Diff5) & (S1 | ~S2 | ~S3 | ~Diff1 | ~Diff2 | ~Diff5) & (
                        S0 | S1 | S2 | ~Diff3 | Diff4 | Diff5) & (S0 | S1 | S2 | ~Diff3 | ~Diff4 | ~Diff5) & (
                        S0 | S1 | S2 | Diff3 | ~Diff4 | Diff5) & (S0 | S1 | S2 | ~S3 | ~Diff2 | Diff3) & (
                        ~S1 | S3 | Diff0 | Diff1 | ~Diff3 | ~Diff4 | ~Diff5) & (
                        S1 | ~S2 | S3 | Diff0 | ~Diff1 | Diff2 | ~Diff4) & (
                        S0 | S1 | S2 | S3 | ~Diff0 | ~Diff1 | Diff2) & (
                        S1 | S3 | Diff0 | Diff1 | Diff2 | Diff3 | Diff4 | Diff5),
                AllOne & NS0 == AllOne & (~Diff0 | Diff5) & (~Diff1 | Diff4) & (~Diff2 | Diff3) & (
                        Diff0 | Diff1 | ~Diff4 | Diff5) & (Diff0 | Diff1 | Diff2 | Diff4) & (
                        Diff0 | ~Diff1 | ~Diff4 | Diff5) & (Diff0 | Diff1 | Diff2 | Diff3) & (
                        Diff1 | ~Diff2 | ~Diff3 | Diff4) & (~Diff0 | Diff2 | Diff3 | ~Diff5) & (
                        Diff0 | Diff1 | Diff2 | ~Diff3 | ~Diff4 | ~Diff5),
                AllOne & NS1 == AllOne & (~Diff0 | Diff5) & (~Diff1 | Diff4) & (~Diff2 | Diff3) & (
                        Diff0 | ~Diff1 | ~Diff4 | ~Diff5) & (Diff1 | ~Diff2 | ~Diff3 | ~Diff4) & (
                        ~Diff0 | Diff2 | ~Diff3 | ~Diff5) & (~Diff0 | ~Diff1 | ~Diff2 | ~Diff3 | ~Diff4 | ~Diff5),
                AllOne & NS2 == AllOne & (Diff3 | Diff4 | Diff5) & (Diff0 | Diff1 | ~Diff4 | Diff5) & (
                        Diff0 | Diff1 | Diff2 | Diff4) & (Diff0 | Diff1 | Diff2 | Diff3) & (
                        Diff0 | ~Diff1 | ~Diff4 | ~Diff5) & (Diff1 | ~Diff2 | ~Diff3 | ~Diff4) & (
                        ~Diff0 | Diff2 | ~Diff3 | ~Diff5) & (~Diff0 | ~Diff1 | ~Diff2 | ~Diff3 | ~Diff4 | ~Diff5),
                AllOne & NS3 == AllOne & (Diff0 | ~Diff1 | ~Diff5) & (~Diff0 | Diff2 | ~Diff3) & (
                        Diff1 | ~Diff2 | ~Diff4) & (Diff0 | ~Diff1 | ~Diff4 | Diff5) & (
                        Diff1 | ~Diff2 | ~Diff3 | Diff4) & (~Diff0 | Diff2 | Diff3 | ~Diff5) & (
                        Diff0 | Diff1 | Diff2 | Diff3 | Diff4 | Diff5) & (
                        S1 | ~Diff1 | ~Diff2 | Diff3 | Diff4 | Diff5),
                AllOne & HWPr == AllOne & (~S0 | S3 | Diff0 | ~Diff1 | ~Diff3) & (
                        ~S0 | S3 | ~Diff0 | Diff1 | ~Diff3) & (~S2 | ~S3 | ~Diff0 | ~Diff2 | Diff4) & (
                        Diff0 | Diff1 | Diff2 | ~Diff3 | ~Diff4 | ~Diff5) & (
                        Diff0 | Diff1 | Diff2 | Diff3 | Diff4 | Diff5) & (
                        ~Diff0 | ~Diff1 | ~Diff2 | ~Diff3 | ~Diff4 | ~Diff5) & (
                        S2 | ~S3 | Diff0 | Diff2 | Diff4 | Diff5) & (
                        S0 | ~Diff0 | ~Diff1 | ~Diff2 | Diff3 | Diff5) & (S2 | ~S3 | Diff1 | Diff2 | Diff3 | Diff5),

            ]
        else:

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
        if MergeTrans:
            Constraints += [
                Flag_Highest == (~S0_Highest | ~S2_Highest) & (~S1_Highest | ~v0Xorv2) & (~S0_Highest | v0Xorv4) & (
                        S3_Highest | ~v0Xorv2 | ~v0Xorv4) & (~S2_Highest | S3_Highest | ~v0Xorv2) & (
                        ~S1_Highest | S3_Highest | ~v0Xorv4) & (S0_Highest | S2_Highest | ~v0Xorv4) & (
                        S1_Highest | ~S2_Highest | ~S3_Highest | ~v0Xorv4) & (
                        S2_Highest | ~S3_Highest | v0Xorv2 | ~v0Xorv4) & (
                        S2_Highest | ~S3_Highest | ~v0Xorv2 | v0Xorv4) & (
                        ~S2_Highest | ~S3_Highest | v0Xorv2 | v0Xorv4) & (
                        S1_Highest | S3_Highest | v0Xorv2 | v0Xorv4),
                HWPr_Highest == 0, ]

        else:
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
            Constraints += [
                (self._P[0][2] ^ NeutralDiff[0]) == 0,
                (self._P[0][3] ^ NeutralDiff[1]) == 0,
            ]

       
        ObjHWDiff = 0
        for i in range(r1):
            ObjHWDiff += self.HW(Pr2[i], self._WordSize)
      
            if AddMatSuiBound:
                if str(self._WordSize * 2) in self._BoundDiffPropagation.keys():
                    Bound = self._BoundDiffPropagation[str(self._WordSize * 2)]
                    Constraints += [ObjHWDiff >= (Bound[i + 1])]

   
        ObjBoomerangSwitch = 0
        for i in range(r1):
            ObjBoomerangSwitch += self.HW(self._Pr[i], self._WordSize)
     
            if AddMatSuiBound:
                if str(self._WordSize * 2) in self._BoundDiffPropagation.keys():
                    Bound = self._BoundDiffPropagation[str(self._WordSize * 2)]
                    Constraints += [ObjBoomerangSwitch >= (Bound[i + 1])]


        if SearchRemaining:
            Constraints += [(-ObjHWDiff + ObjBoomerangSwitch) >= HWPr]
        else:
            Constraints += [(-ObjHWDiff + ObjBoomerangSwitch) == HWPr]

 
        for i in range(len(ExcludeNB)):
            InDiffL = self._P[0][2]
            InDiffR = self._P[0][3]
            Impossible = ExcludeNB[i]
            Constraints += [((InDiffL ^ Impossible[0]) | (InDiffR ^ Impossible[1])) >= 1]
     

        if ExcludeNBMask != []:
            for i in range(len(ExcludeNBMask)):
                InDiffL = self._P[0][2]
                InDiffR = self._P[0][3]
                MASK = ExcludeNBMask[i]

                if i == 0:
                    SUM = self.HW(((InDiffL & MASK[0]) ^ (InDiffR & MASK[1])), self._WordSize) & 1
                else:
                    SUM = SUM | (self.HW(((InDiffL & MASK[0]) ^ (InDiffR & MASK[1])), self._WordSize) & 1)
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
            print("The maximum Hamming weight of a neutral bit:", MaxHW_NB)
            InDiffL = self._P[0][2]
            InDiffR = self._P[0][3]
            Constraints += [self.HW(InDiffL, self._WordSize) + self.HW(InDiffR, self._WordSize) <= MaxHW_NB]

        if MinHW_NB > 0:
            print("The minimum Hamming weight of a neutral bit:", MinHW_NB)
            InDiffL = self._P[0][2]
            InDiffR = self._P[0][3]
            Constraints += [self.HW(InDiffL, self._WordSize) + self.HW(InDiffR, self._WordSize) >= MinHW_NB]

     
        for i in Constraints:
            self._Solver.add(i)

        Result = self._Solver.check()

        model = {}
        if Result:  
            model = self._Solver.model()
      
        return Result, model



def GetP(StartRound, R, nrounds):
    r0 = nrounds[0]
    r1 = nrounds[1]
    r2 = nrounds[2]
    if (R < r0) | (R > (r0 + r1)):
        P = ["p" + str(StartRound + R) + "L", "p" + str(StartRound + R) + "R"]
    elif r1 == 0:
        P = ["p" + str(StartRound + R) + "L", "p" + str(StartRound + R) + "R"]
    else:
        P = ["p" + str(StartRound + R) + "_" + str(i) for i in range(4)]
    return P


def GetPr(StartRound, R, Suffix="", VarName="pr"):
    Pr = [VarName + str(StartRound + R) + Suffix]
    return Pr


def GetK(R, nrounds, VarName="k"):


    r0 = nrounds[0]
    r1 = nrounds[1]
    if (R < (r0 + max(0, r1 - 1))):
        K = [VarName + str(R) + "L", VarName + str(R) + "R"]
    else:
        K = [VarName + str(R) + "L", ]

    return K


def DisplayDiffPropagation(model, nrounds,
                           m=4, WordSize=16, FileName=None, RelatedKey=False,
                           NoPr_BoomerangSwitch=False):
    StartRound = 0
    R = nrounds[0] + nrounds[1] + nrounds[2]
    P = [GetP(StartRound, i, nrounds) for i in range(R + 1)]
    Pr = [GetPr(StartRound, i) for i in range(R)]

    K = [GetK(i, nrounds, VarName="k") for i in range(nrounds[0] + m - 2 + max(0, nrounds[1] - 1))]
    K_Pr = [GetPr(0, i, VarName="k_pr") for i in range(nrounds[0] - 1 + max(0, nrounds[1] - 1))]


    HWPr = 0
    for i in range(R):
        if (i >= nrounds[0]) & (i < (nrounds[0] + nrounds[1])):
            if NoPr_BoomerangSwitch:
                continue
            else:
                HWPr += hw(model[Pr[i][0]], WordSize)
        else:
            HWPr += 2 * hw(model[Pr[i][0]], WordSize)


    BoomerangCharactistic = []
    BoomerangInDiff = [0, 0]
    BoomerangOutDiff = [0, 0]

    if nrounds[1] > 0:

    
        BoomerangInDiff = [model[P[nrounds[0]][0]], model[P[nrounds[0]][1]], ]
        BoomerangOutDiff = [model[P[nrounds[0] + nrounds[1]][2]], model[P[nrounds[0] + nrounds[1]][3]], ]


        if nrounds[1] == 1:
       
            Diff = [[model[P[nrounds[0]][0]], model[P[nrounds[0]][1]], ],
                    [],
                    [model[P[nrounds[0] + 1][2]], model[P[nrounds[0] + 1][3]], ],
                    []]
            RealPr_Switch = Matrice.ComputeBCT_SPECKOneRound(WordSize, Diff)
            if (model[P[nrounds[0]][0]] | model[P[nrounds[0]][1]]) == 0:
                HWPr_Switch = 0
            else:
                HWPr_Switch = -mt.log2(RealPr_Switch)


        else: 
            HWPr_Switch = 0
         

            if not RelatedKey:
                Diff = [[model[P[nrounds[0]][0]], model[P[nrounds[0]][1]], ],
                        [model[P[nrounds[0] + 1][0]], model[P[nrounds[0] + 1][1]], ],
                        [model[P[nrounds[0] + 1][2]], model[P[nrounds[0] + 1][3]], ],
                        [0, 0]]
            else:
                Diff = [[model[P[nrounds[0]][0]], model[P[nrounds[0]][1]], ],
                        [model[P[nrounds[0] + 1][0]] ^ model[K[nrounds[0]][1]],
                         model[P[nrounds[0] + 1][1]] ^ model[K[nrounds[0]][1]]],
                        [model[P[nrounds[0] + 1][2]] ^ model[K[nrounds[0]][1]],
                         model[P[nrounds[0] + 1][3]] ^ model[K[nrounds[0]][1]]],
                        [0, 0]]
            RealPr_UBCT = Matrice.ComputeUBCT_SPECKOneRound(WordSize, Diff)
            if sum(Diff[0]) == 0:
                HWPr_Switch += 0
            else:
                HWPr_Switch += -mt.log2(RealPr_UBCT)

         
            temp_R = nrounds[0] + nrounds[1]
            Diff = [[model[P[temp_R - 1][0]], model[P[temp_R - 1][1]], ],
                    [0, 0],
                    [model[P[temp_R][2]], model[P[temp_R][3]]],
                    [model[P[temp_R - 1][2]], model[P[temp_R - 1][3]], ]]

            RealPr_LBCT = Matrice.ComputeLBCT_SPECKOneRound(WordSize, Diff)

            if sum(Diff[2]) == 0:
                HWPr_Switch += 0
            else:
                HWPr_Switch += -mt.log2(RealPr_LBCT)

    

        if nrounds[1] == 1:
            Counter = hw(model[Pr[nrounds[0]][0]], WordSize)
        else:
            Counter = hw(model[Pr[nrounds[0]][0]], WordSize)
            Counter = Counter + hw(model[Pr[nrounds[0] + nrounds[1] - 1][0]], WordSize)

   
        if NoPr_BoomerangSwitch:
            RealHWPr = HWPr + HWPr_Switch
            if nrounds[1] > 2:
                for i in range(nrounds[0] + 1, nrounds[0] + nrounds[1] - 1):
                    RealHWPr += hw(model[Pr[i][0]], WordSize)

        else:
            RealHWPr = HWPr - Counter + HWPr_Switch
    else:
        RealHWPr = HWPr
    print(str(R) + "-round, the preliminary estimate of the differential weight :\t", HWPr)
    print(str(R) + "-round, the expected differential weight:\t", RealHWPr)
    print("The starting round is:\t", str(StartRound))

    print("**********************")
    print("Differential propagation")
    for i in range(R + 1):
        for j in range(len(P[i])):
            temp = "{:0" + str(WordSize) + "b}"
            print(temp.format(model[P[i][j]]), end="\t")
        print()

    for i in range(nrounds[0] + 1):
        for j in range(2):
            BoomerangCharactistic.append(model[P[i][j]])

    for i in range(nrounds[1] - 1):
        for j in range(4):
            BoomerangCharactistic.append(model[P[nrounds[0] + 1 + i][j]])


    for i in range(nrounds[0] + nrounds[1], R + 1):
        for j in range(2):
            BoomerangCharactistic.append(model[P[i][-2 + j]])

    if RelatedKey:  
    
        for i in range(nrounds[0] + m - 2 + nrounds[1] - 1):
            for j in range(len(K[i])):
                BoomerangCharactistic.append(model[K[i][j]])

    print("**********************")
    print("Corresponding probability")
    for i in range(R):
        for j in range(len(Pr[i])):
            temp = "{:0" + str(WordSize) + "b}"
            print(temp.format(model[Pr[i][j]]), end="\t")
        print()


    temp = "{:0" + str(WordSize) + "b}"
    for i in range(nrounds[1]):
        print("Em's", i, "-round")
        for j in range(4):
            StateName = "S" + str(j) + "_R" + str(i + nrounds[0])
            print(StateName, temp.format(model[StateName]))
        FlagName = "Flag_R" + str(i + nrounds[0])
        print(FlagName, temp.format(model[FlagName]))

    if RelatedKey:  

        print("**********************")
        print("Differential propagation on keys")
        for i in range(nrounds[0] + m - 2 + max(0, nrounds[1] - 1)):
            for j in range(len(K[i])):
                temp = "{:0" + str(WordSize) + "b}"
                print(temp.format(model[K[i][j]]), end="\t")
            print()

        print("**********************")
        print("The probability corresponding to differential propagation on the key")
        Pr_K = 0
        for i in range(nrounds[0] - 1 + max(0, nrounds[1] - 1)):
            for j in range(len(K_Pr[i])):
                temp = "{:0" + str(WordSize) + "b}"
                print(temp.format(model[K_Pr[i][j]]), end="\t")
                Pr_K += hw(model[K_Pr[i][j]], WordSize)
            print()
        print("The probability weight of differential propagation on the key is", Pr_K)

    
    temp = "0x{:0" + str(int(WordSize / 4)) + "x}"
    print(hex(BoomerangInDiff[0]), hex(BoomerangInDiff[1]), )
    print(hex(BoomerangOutDiff[0]), hex(BoomerangOutDiff[1]), )
    if RelatedKey:
        KeyDiff = [temp.format(model[K[0][1]]), Pr_K]
        for i in range(m - 1):
            KeyDiff = [temp.format(model[K[i][0]])] + KeyDiff

        return HWPr, [model[P[0][0]], model[P[0][1]], model[P[-1][-2]], model[P[-1][-1]],
                      KeyDiff,
                      [temp.format(BoomerangInDiff[0]), temp.format(BoomerangInDiff[1]),
                       temp.format(BoomerangOutDiff[0]),
                       temp.format(BoomerangOutDiff[1]), RealHWPr], BoomerangCharactistic
                      ]
    else:
        return HWPr, [model[P[0][0]], model[P[0][1]], model[P[-1][-2]], model[P[-1][-1]],
                      [temp.format(BoomerangInDiff[0]), temp.format(BoomerangInDiff[1]),
                       temp.format(BoomerangOutDiff[0]),
                       temp.format(BoomerangOutDiff[1]), RealHWPr], BoomerangCharactistic
                      ]


def DisplayDiffPropagation_NB(model, HWPr, nrounds, WordSize=16, ReturnTrail=False):
    StartRound = 0
    R = nrounds[0] + nrounds[1] + nrounds[2]
    print(str(R) + "-round,  the differential weight:\t", HWPr)
    print("The starting round is:\t", str(StartRound))
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

    if ReturnTrail:
        #
        Trail = []
        for i in P:
            for j in i:
                Trail.append(model[j])
        Trail.append(HWPr)

        return [model[P[0][2]], model[P[0][3]]], Trail
    else:
        return [model[P[0][2]], model[P[0][3]]]


def DisplayDiffPropagation_NB_Method2(model, HWPr, nrounds, WordSize=16):
    StartRound = 0
    R = nrounds[0] + nrounds[1] + nrounds[2]
    print(str(R) + "-round cipher, the differential weight:\t", HWPr)
    print("The starting round is:\t", str(StartRound))
    P = [GetP(StartRound, i, nrounds) for i in range(R + 1)]
    Pr = [GetPr(StartRound, i, ) for i in range(R)]

    Pr2 = [GetPr(StartRound, i, Suffix="_Diff") for i in range(1)]
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
    for i in range(1):
        for j in range(len(Pr2[i])):
            temp = "{:0" + str(WordSize) + "b}"
            print(temp.format(model[Pr2[i][j]]), end="\t")
        print()

    return [model[P[0][2]], model[P[0][3]]]







def SearchGoodNB(nrounds, DiffPropa, WordSize=16, AddMatSuiBound=False, ExcludeNB=[], ExcludeNBMask=[], MinHWPr=-1,
                 MaxHW_NB=-1, MinHW_NB=-1, UpperBoundHWPr=100):
    HWPr = max(0, MinHWPr)
    Flag = True

    StartTime = time.time()
    SearchRemaining = False
    while (Flag):
        print("SPECK", WordSize * 2, "'s structure is:\t", nrounds)
        print("The target value for probability weight is", HWPr)
        DiffModel = SPECK(nrounds=nrounds, WordSize=WordSize)
        Result, model = DiffModel.SearchNB_EBCT(DiffPropa, HWPr, AddMatSuiBound, ExcludeNB, ExcludeNBMask, MaxHW_NB,
                                                MinHW_NB, SearchRemaining=SearchRemaining)
        print(Result)
        if Result:
            print("All variables have the following values")
            print(model)
            ValidNB = DisplayDiffPropagation_NB(model=model, HWPr=HWPr, nrounds=nrounds, WordSize=WordSize)
            Flag = False
        HWPr += 1
        EndTime = time.time()
        print("The time used is", EndTime - StartTime, "s")

        print(HWPr)
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

    ValidNB.append(HWPr - 1)
    return True, ValidNB


def SearchGoodNBTrails(nrounds, DiffPropa, NeutralDiff, WordSize=16, AddMatSuiBound=False, ExcludeTrails=[], MinHWPr=-1,
                       MaxHW_NB=-1, MinHW_NB=-1, UpperBoundHWPr=100):
    HWPr = max(0, MinHWPr)
    Flag = True

    StartTime = time.time()
    SearchRemaining = False
    while (Flag):
        print("SPECK", WordSize * 2, "'s structure :\t", nrounds)
        print("The target value for probability weight is", HWPr)
        DiffModel = SPECK(nrounds=nrounds, WordSize=WordSize)
        Result, model = DiffModel.SearchNB_EBCT(DiffPropa, HWPr, AddMatSuiBound, MaxHW_NB=MaxHW_NB,
                                                MinHW_NB=MinHW_NB, SearchRemaining=SearchRemaining,
                                                ExcludeTrails=ExcludeTrails, NeutralDiff=NeutralDiff
                                                )
        print(Result)
        if Result:
            print("All variables have the following values")
            print(model)
            ValidNB, Trail = DisplayDiffPropagation_NB(model=model, HWPr=HWPr, nrounds=nrounds, WordSize=WordSize,
                                                       ReturnTrail=True)
            Flag = False
        HWPr += 1
        EndTime = time.time()
        print("The time used is", EndTime - StartTime, "s")

        if HWPr > UpperBoundHWPr:
            if Flag == False:  
                print("When HWPr>=", UpperBoundHWPr, "There is still a solution, but we will no longer search!")
                return False, ValidNB, Trail
            elif HWPr == (UpperBoundHWPr + 1):
                SearchRemaining = True
            else:
                print("******************")
                print("There is no solution in the remaining space!")
                print("******************")
                return False, [], None

    ValidNB.append(HWPr - 1)
    return True, ValidNB, Trail



def hw(X, WordSize=16):
    Counter = 0
    for i in range(WordSize):
        if (X >> i) & 1 == 1:
            Counter += 1
    return Counter


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


def GenBaseVector(ObtainedVector, n):
    BaseVector = ObtainedVector.copy()

    for i in range(n):
        OneVector = np.zeros((1, n), dtype=np.bool_)
        OneVector[0, n - i - 1] = 1

        temp = np.concatenate((BaseVector, OneVector))
        if np.linalg.matrix_rank(temp) > np.linalg.matrix_rank(BaseVector):
            BaseVector = temp.copy()
            if np.linalg.matrix_rank(temp) == n:
                BaseVector = BaseVector.T
                Inv_BaseVector = np.linalg.inv(BaseVector) % 2
                Inv_BaseVector = np.array(Inv_BaseVector, dtype=np.int)
                print("!!!!!!!!!!! 1 !!!!!!!!!!!")
                print(np.matmul(BaseVector, Inv_BaseVector) % 2)
                return BaseVector, Inv_BaseVector

    while (np.linalg.matrix_rank(BaseVector) < n):
        OneVector = np.random.randint(2, size=(1, n))
        temp = np.concatenate((BaseVector, OneVector))
        if np.linalg.matrix_rank(temp) > np.linalg.matrix_rank(BaseVector):
            BaseVector = temp.copy()

    BaseVector = BaseVector.T

    Inv_BaseVector = np.linalg.inv(BaseVector) % 2
    Inv_BaseVector = np.array(Inv_BaseVector)
    print("!!!!!!!!!!!2 !!!!!!!!!!!")
    print(np.matmul(BaseVector, Inv_BaseVector) % 2)
    return BaseVector, Inv_BaseVector


def GenBaseVector2(ObtainedVector, n):
    BaseVector = ObtainedVector.copy()

    assert len(BaseVector) == bit_rank(BaseVector, len(BaseVector), n), "The rows of the input matrix are not linearly independent!" + str(
        len(BaseVector)) + str(bit_rank(BaseVector, len(BaseVector), n))
    for i in range(n):
        OneVector = np.zeros((1, n), dtype=np.int_)
        OneVector[0, n - i - 1] = 1

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


def Mask2Bin(dl, dr, n):
    Vector = np.zeros((1, n), dtype=np.int_)
    for i in range(n):
        if i < (n / 2):
            Vector[0, i] = int(dl >> int(n / 2 - 1 - i)) & 1
        else:
            Vector[0, i] = int(dr >> int(n - 1 - i)) & 1
    return Vector


def Bin2Mask(BinNum, n):
    d = 0
    for i in range(n):
        if BinNum[-1 - i]:
            d += 1 << i
    LenMask = int(n / 2)
    dl = d >> LenMask
    dr = d & (2 ** LenMask - 1)
    return dl, dr


def ExcludeNB_Mask( NB_Obtained, WordSize):
 
    ObtainedVectors = Mask2Bin(NB_Obtained[0][0], NB_Obtained[0][1], WordSize * 2)
 
    for NB in NB_Obtained[1:]:
        temp = Mask2Bin(NB[0], NB[1], WordSize * 2)
        ObtainedVectors = np.concatenate((ObtainedVectors, temp))

    BaseVectors, Inv_BaseVectors = GenBaseVector2(ObtainedVectors, WordSize * 2)

    
    ExcludeMask = []
    for i in range(len(NB_Obtained) , WordSize * 2):
        dl, dr = Bin2Mask(Inv_BaseVectors[i, :], WordSize * 2)
        ExcludeMask.append([dl, dr])

    return ExcludeMask


def modify_matrix(matrix):
    rows, cols = matrix.shape

    Num=0
    for j in range(cols):
        i = Num
        while ((i < rows) and (matrix[i, j] == 0)):
          
            i += 1
        if i == rows:
            continue

        

        if i!=Num:
            temp=matrix[Num].copy()
            matrix[Num]=matrix[i]
            matrix[i]=temp

  
        for y in range(rows):
            if y==Num:
                continue
            if matrix[y,j]==1:
                matrix[y]^=matrix[Num]
        Num += 1
  
    return matrix


if __name__ == "__main__":
    
    StartTime = time.time()
    SearchDiff = False
    SearchNeuralBitIterative = True
    SearchNeuralBitSpecifiedDiff = False
    GenBasis = False

    if SearchDiff:
 
        SearchNum = 2 ** 6
        SpecifyDiff = [[], [0x2800, 0x0010]]

        DiffPropaList = []
        nrounds = [3, 0, 0]
        HWPr = 0
        WordSize = 16
        AddMatSuiBound = False
        while (len(DiffPropaList) < SearchNum):

            StartTime = time.time()
            DiffModel = SPECK(nrounds=nrounds, WordSize=WordSize)
            Result, model = DiffModel.SearchDiff(HWPr, AddMatSuiBound, ExcludePoint=DiffPropaList,
                                                 SpecifyDiff=SpecifyDiff)
            print(HWPr, Result)
            if Result:
                print("All variables have the following values")
                print(model)
                InDiffL, InDiffR, OutDiffL, OutDiffR, HWPr = DisplayDiffPropagation(model=model, HWPr=HWPr,
                                                                                    nrounds=nrounds, WordSize=WordSize)
                temp = "{:0" + str(int(WordSize / 4)) + "x}"
                DiffPropaList.append([InDiffL, InDiffR, OutDiffL, OutDiffR, HWPr])
                EndTime = time.time()
                print("The time used is ", EndTime - StartTime, "s")
                StartTime = time.time()
            else:
                HWPr += 1

        print( SearchNum, " trails found")
        tempDiffPropaList = [["0x" + temp.format(i[0]), "0x" + temp.format(i[1]),
                              "0x" + temp.format(i[2]), "0x" + temp.format(i[3]),
                              i[4]] for i in DiffPropaList]
        print(tempDiffPropaList)

    if SearchNeuralBitIterative:
   

        Method_ExcludeNB = 1  

        SearchNum = 2 ** 5
       

     

        nrounds = [0, 2, 0]
        DiffPropa =[0x2a10, 0x0004, 0x8000, 0x0100]

       

        WordSize = 16
        MaxHW_NB = -1
        MinHW_NB = -1

        AddMatSuiBound = True
        NB_Obtained = [


        ]

        if len(DiffPropa) != 0:
            NB_Obtained += [[DiffPropa[0], DiffPropa[1], 0, 0]]  

        ExcludeNB = []  
        ExcludeNB += NB_Obtained

        while (len(NB_Obtained) < SearchNum):
            StartAllTime = time.time()
            MinHWPr=-1
            if len(NB_Obtained) >1 :
                MinHWPr = NB_Obtained[-1][-2]
            else:
                MinHWPr = 0
            ExcludeNBMask = []
            if Method_ExcludeNB != 0:
                ExcludeNBMask = ExcludeNB_Mask( NB_Obtained, WordSize)
                print("ExcludeNBMask:")
                temp = [["{:04x}".format(i[0]), "{:04x}".format(i[1])] for i in ExcludeNBMask]
                print(temp)
                ExcludeNB_Num = 1
            else:
                ExcludeNB_Num = len(ExcludeNB)
            FlagContinue, ValidNB = SearchGoodNB(nrounds, WordSize=WordSize, AddMatSuiBound=AddMatSuiBound,
                                                 DiffPropa=DiffPropa, ExcludeNB=ExcludeNB[:ExcludeNB_Num],
                                                 ExcludeNBMask=ExcludeNBMask, MinHWPr=MinHWPr,
                                                 MaxHW_NB=MaxHW_NB, MinHW_NB=MinHW_NB)

            EndAllTime = time.time()
            print("The time used is", EndAllTime - StartAllTime, "s")
            ValidNB.append(EndAllTime - StartAllTime)

            if Method_ExcludeNB != 0:
                if FlagContinue:
                    NB_Obtained.append(ValidNB)
            else:
                Flag, ExcludeNB, ValidNB = AddExcludePoint(ExcludeNB, ValidNB)
                if Flag:
                    NB_Obtained.append(ValidNB)

         
            temp = "{:0" + str(int(WordSize / 4)) + "x}"
            print("NB_Obtained length", len(NB_Obtained))
            temp1 = [[i[j] if j > 1 else "0x" + temp.format(i[j]) for j in range(len(i))] for i in NB_Obtained[:]]
            print(temp1)
            if not FlagContinue:
                print("There is no way to search for results again!")
                break

        print("***************************")
        print("Result:")

        temp = "{:0" + str(int(WordSize / 4)) + "x}"
        print("ExcludeNB length:", len(ExcludeNB))

        print("NB_Obtained length:", len(NB_Obtained))
        NB_Obtained = [[i[j] if j > 1 else "0x" + temp.format(i[j]) for j in range(len(i))] for i in NB_Obtained[:]]
        print(NB_Obtained)

    if SearchNeuralBitSpecifiedDiff:


        Method_ExcludeNB = 1  

        SearchTimes = 2 ** 8

       

        nrounds = [0, 2, 0]
        DiffPropa = [0x2a10, 0x0004, 0x8000, 0x0100]
        NeutralDiff = [0x2e10,0x0004,]

      
        WordSize = 16
        MaxHW_NB = -1
        MinHW_NB = -1

        AddMatSuiBound = True

        ExcludeTrails = [] 
        NB_Obtained = []

        for Index in range(SearchTimes):

            StartAllTime = time.time()
            if len(ExcludeTrails) != 0:
                MinHWPr = ExcludeTrails[-1][-1]
            else:
                MinHWPr = 0

            FlagContinue, ValidNB, Trail = SearchGoodNBTrails(nrounds, WordSize=WordSize, AddMatSuiBound=AddMatSuiBound,
                                                              DiffPropa=DiffPropa, NeutralDiff=NeutralDiff,
                                                              ExcludeTrails=ExcludeTrails,
                                                              MinHWPr=MinHWPr,
                                                              MaxHW_NB=MaxHW_NB, MinHW_NB=MinHW_NB)

            if Trail != None:
                ExcludeTrails.append(Trail)
            EndAllTime = time.time()
            print("The time used is", EndAllTime - StartAllTime, "s")
            if ValidNB != []:
                ValidNB.append(EndAllTime - StartAllTime)
                NB_Obtained.append(ValidNB)

    
            temp = "{:0" + str(int(WordSize / 4)) + "x}"
            print("ExcludeTrails length:", len(ExcludeTrails))

            if not FlagContinue:
                print("There is no way to search for results again!")
                break

        print("***************************")
        print("Result:")

        temp = "{:0" + str(int(WordSize / 4)) + "x}"
        print("ExcludeTrails :", len(ExcludeTrails))
        print(ExcludeTrails)
        STR_NB_Obtained = [[i[j] if j > 1 else "0x" + temp.format(i[j]) for j in range(4)] for i in NB_Obtained[:]]
        print("***************************")
        print("The probability weights of the obtained neutral bits are")
        print(STR_NB_Obtained)

        print("***************************")

        SumPr = 0
        for i in NB_Obtained:
            SumPr += 2 ** (-i[-2])
        print(len(ExcludeTrails), " trails' probability", SumPr)
        if SumPr > 0:
            print("The probability weight is", -mt.log2(SumPr))


        ObtainedLinearIndependentBases = [
            # 0x00100000,
            # 0x00200000,
            # 0x00400000,
            # 0x00010200,
            # 0x00004000,
            # 0x00008000,
            # 0x20000040,
            # 0x40000000,
            # 0x00800000,
            # 0x00000001,
            # 0x00000080,
            # 0x08000800,
            # 0x03518a04,
            # 0x82110b04,
            # 0x00000100,
            # 0x28000010,
            # 0x01000002,
            # 0x00020400,
            # 0x02797a04,
            # 0x10000020,
            # 0x38000010,
            # 0x00081000,
            # 0x1a110a14,
            # 0x06110a04,
            # 0x80020100,
            # 0x00080000,
            # 0x0c000008,
            # 0x04000008,
            # 0x000e0400,
            # 0x02000804,


            # 0x28000010,
            # 0xa8400010,
            # 0x29408010,
            # 0xa8410010,
            # 0x81408100,
            # 0xa8430010,
            # 0x83408100,
            # 0x28050210,
            # 0xa9502010,
            # 0x87408100,
            # 0x20540800,
            # 0xa9040512,
            # 0x280a0410,
            # 0x81080502,
            # 0x68a04090,
            # 0x00140800,
            # 0x2a810211,
            # 0x2b810211,
            # 0x28740810,
            # 0x6a050090,
            # 0x788000f0,
            # 0x00340800,
            # 0x001c0800,
            # 0x50000020,
            # 0x50204020,
            # 0x287c1810,
            # 0x68287010,
            # 0x3e000014,
            # 0x14081008,
            # 0x280c0810,
            # 0x040a0408,
            # 0x00000010,

            0x28000010,
            0x81408100,
            0xa8400010,
            0x00140800,
            0x83408100,
            0xa8410010,
            0x20540800,
            0x280a0410,
            0x001c0800,
            0x280c0810,
            0x2a810211,
            0x81080502,
            0x29408010,
            0x87408100,
            0xa8430010,
            0x00340800,
            0x28740810,
            0xa9502010,
            0xa9040512,
            0x040a0408,
            0x6a050090,
            0x28050210,
            0x3e000014,
            0x50000020,
            0x2b810211,
            0x68a04090,
            0x14081008,
            0x50204020,
            0x287c1810,
            0x68287010,
            0x788000f0,
            0x00000010,

        ]
        WordSize = 16

        # 生成基转换的矩阵
        for i in range(len(ObtainedLinearIndependentBases)):
            NB = ObtainedLinearIndependentBases[i]
            if i == 0:
                ObtainedVectors = Mask2Bin(NB >> 16, NB & 0xffff, WordSize * 2)
            else:
                temp = Mask2Bin(NB >> 16, NB & 0xffff, WordSize * 2)
                ObtainedVectors = np.concatenate((ObtainedVectors, temp))

        BaseVectors, Inv_BaseVectors = GenBaseVector2(ObtainedVectors, WordSize * 2)

        print("******************************")
        print("BaseVectors.T")
        print(BaseVectors)

        print("BaseVectors")
        print(BaseVectors.T)

        print("下面生成这个向量矩阵的最简阶梯形")
        matrix = np.array(BaseVectors.T, dtype=int)
        print(matrix)
        # 修改矩阵
        modified_matrix = modify_matrix(matrix)
        # 输出结果
        print("修改后的矩阵：")
        print(modified_matrix)



