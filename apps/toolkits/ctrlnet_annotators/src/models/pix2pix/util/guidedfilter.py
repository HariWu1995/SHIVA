import numpy as np


class GuidedFilter():

    def __init__(self, source, reference, r=64, eps=0.05**2):
        self.source = source
        self.reference = reference
        self.r = r
        self.eps = eps
        self.smooth = self.guided_filter(self.source, self.reference, self.r, self.eps)

    def box_filter(self,img, r):
        (rows, cols) = img.shape
        imDst = np.zeros_like(img)

        imCum = np.cumsum(img, 0)
        imDst[0 : r+1, :] = imCum[r : 2*r+1, :]
        imDst[r+1 : rows-r, :] = imCum[2*r+1 : rows, :] - imCum[0 : rows-2*r-1, :]
        imDst[rows-r: rows, :] = np.tile(imCum[rows-1, :], [r, 1]) - imCum[rows-2*r-1 : rows-r-1, :]

        imCum = np.cumsum(imDst, 1)
        imDst[:, 0 : r+1] = imCum[:, r : 2*r+1]
        imDst[:, r+1 : cols-r] = imCum[:, 2*r+1 : cols] - imCum[:, 0 : cols-2*r-1]
        imDst[:, cols-r: cols] = np.tile(imCum[:, cols-1], [r, 1]).T - imCum[:, cols-2*r-1 : cols-r-1]

        return imDst

    def guided_filter(self, I, p, r, eps):
        (rows, cols) = I.shape
        N = self.box_filter(np.ones([rows, cols]), r)

        meanI  = self.box_filter(    I, r) / N
        meanP  = self.box_filter(    p, r) / N
        meanIp = self.box_filter(I * p, r) / N
        meanII = self.box_filter(I * I, r) / N

        covIp = meanIp - meanI * meanP
        varI  = meanII - meanI * meanI

        a = covIp / (varI + eps)
        b = meanP - a * meanI

        meanA = self.box_filter(a, r) / N
        meanB = self.box_filter(b, r) / N

        q = meanA * I + meanB
        return q

