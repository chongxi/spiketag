import numpy as np
from ..base import probe
from ..view import probe_view

prb_bowtie_L = probe(shank_no=3, prb_type='bow_tie')

prb_bowtie_L.shanks[0].l = [59,60,10,58,12,11,57,56]
prb_bowtie_L.shanks[0].r = [5,52,3,54,53,4,13,2,55]
prb_bowtie_L.shanks[0].xl = -100.
prb_bowtie_L.shanks[0].yl = 20
prb_bowtie_L.shanks[0].xr = -80.
prb_bowtie_L.shanks[0].yr = 5

prb_bowtie_L.shanks[1].l = [15,63,48,47,0,61,9,14,62,6]
prb_bowtie_L.shanks[1].r = [8, 1,51,50,18,34,31,25,33,17,22,49]
prb_bowtie_L.shanks[1].xl = -10.
prb_bowtie_L.shanks[1].yl = 15
prb_bowtie_L.shanks[1].xr = 10.
prb_bowtie_L.shanks[1].yr = 0 

prb_bowtie_L.shanks[2].l = [39,38,20,45,44,24,7,32,16,23,46,30]
prb_bowtie_L.shanks[2].r = [19,37,21,35,36,26,29,40,27,42,41,28,43]
prb_bowtie_L.shanks[2].xl = 80.
prb_bowtie_L.shanks[2].yl = 10 
prb_bowtie_L.shanks[2].xr = 100.
prb_bowtie_L.shanks[2].yr = -5
prb_bowtie_L.auto_pos()
prb_bowtie_L.mapping[5]  += np.array([-10,2])
prb_bowtie_L.mapping[52] += np.array([-2, 0])
prb_bowtie_L.mapping[8]  += np.array([-10,2])
prb_bowtie_L.mapping[1]  += np.array([-2, 0])
prb_bowtie_L.mapping[19] += np.array([-10,2])
prb_bowtie_L.mapping[37] += np.array([-2, 0])

prb_bowtie_L[0] = np.array([59,  5, 52,  3])
prb_bowtie_L[1] = np.array([60, 10, 54, 53])
prb_bowtie_L[2] = np.array([58, 12,  4, 13])
prb_bowtie_L[3] = np.array([11, 57,  2, 55])

prb_bowtie_L[4] = np.array([15,  8,  1, 51])
prb_bowtie_L[5] = np.array([63, 48, 50, 18])
prb_bowtie_L[6] = np.array([47,  0, 34, 31])
prb_bowtie_L[7] = np.array([61,  9, 25, 33])
prb_bowtie_L[8] = np.array([14, 62, 22, 17])

prb_bowtie_L[9]  = np.array([39, 19, 37, 21])
prb_bowtie_L[10] = np.array([38, 20, 35, 36])
prb_bowtie_L[11] = np.array([45, 44, 26, 29])
prb_bowtie_L[12] = np.array([24,  7, 40, 27])
prb_bowtie_L[13] = np.array([32, 16, 42, 41])
prb_bowtie_L[14] = np.array([23, 46, 28, 43])

# prb_bowtie_L.n_ch = 160


if __name__ == '__main__':
    prb = prb_bowtie_L
    prb_view = probe_view()
    prb_view.set_data(prb)
    prb_view.run()