import numpy as np
from ..base import probe
from ..view import probe_view


###################################################
# bowtie_L
###################################################

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


###################################################
# bowtie_LL
###################################################

prb_bowtie_LL = probe(shank_no=6, prb_type='bow_tie')

prb_bowtie_LL.shanks[0].l = [59,60,10,58,12,11,57,56]
prb_bowtie_LL.shanks[0].r = [5,52,3,54,53,4,13,2,55]
prb_bowtie_LL.shanks[0].xl = -100.
prb_bowtie_LL.shanks[0].yl = 20
prb_bowtie_LL.shanks[0].xr = -80.
prb_bowtie_LL.shanks[0].yr = 5

prb_bowtie_LL.shanks[1].l = [15,63,48,47,0,61,9,14,62,6]
prb_bowtie_LL.shanks[1].r = [8, 1,51,50,18,34,31,25,33,17,22,49]
prb_bowtie_LL.shanks[1].xl = -10.
prb_bowtie_LL.shanks[1].yl = 15
prb_bowtie_LL.shanks[1].xr = 10.
prb_bowtie_LL.shanks[1].yr = 0 

prb_bowtie_LL.shanks[2].l = [39,38,20,45,44,24,7,32,16,23,46,30]
prb_bowtie_LL.shanks[2].r = [19,37,21,35,36,26,29,40,27,42,41,28,43]
prb_bowtie_LL.shanks[2].xl = 80.
prb_bowtie_LL.shanks[2].yl = 10 
prb_bowtie_LL.shanks[2].xr = 100.
prb_bowtie_LL.shanks[2].yr = -5

prb_bowtie_LL.shanks[3].l = [123,124,74,122,76,75,121,120]
prb_bowtie_LL.shanks[3].r = [69,116,67,118,117,68,77,66,119]
prb_bowtie_LL.shanks[3].xl = 350
prb_bowtie_LL.shanks[3].yl = 10
prb_bowtie_LL.shanks[3].xr = 370.
prb_bowtie_LL.shanks[3].yr = -5

prb_bowtie_LL.shanks[4].l = [79,127,112,111,64,125,73,78,126,70]
prb_bowtie_LL.shanks[4].r = [72,65,115,114,82,98,95,89,97,81,86,113]
prb_bowtie_LL.shanks[4].xl = 260.
prb_bowtie_LL.shanks[4].yl = 15
prb_bowtie_LL.shanks[4].xr = 280.
prb_bowtie_LL.shanks[4].yr = 0 

prb_bowtie_LL.shanks[5].l = [103,102,84,109,108,88,71,96,80,87,110,94]
prb_bowtie_LL.shanks[5].r = [83,101,85,99,100,90,93,104,91,106,105,92,107]
prb_bowtie_LL.shanks[5].xl = 170.
prb_bowtie_LL.shanks[5].yl = 20
prb_bowtie_LL.shanks[5].xr = 190.
prb_bowtie_LL.shanks[5].yr = 5

prb_bowtie_LL.auto_pos()
prb_bowtie_LL.mapping[5]  += np.array([-10,2])
prb_bowtie_LL.mapping[52] += np.array([-2, 0])
prb_bowtie_LL.mapping[8]  += np.array([-10,2])
prb_bowtie_LL.mapping[1]  += np.array([-2, 0])
prb_bowtie_LL.mapping[19] += np.array([-10,2])
prb_bowtie_LL.mapping[37] += np.array([-2, 0])
prb_bowtie_LL.mapping[69]  += np.array([-10,2])
prb_bowtie_LL.mapping[116] += np.array([-2, 0])
prb_bowtie_LL.mapping[72]  += np.array([-10,2])
prb_bowtie_LL.mapping[65]  += np.array([-2, 0])
prb_bowtie_LL.mapping[83] += np.array([-10,2])
prb_bowtie_LL.mapping[101] += np.array([-2, 0])

prb_bowtie_LL[0] = np.array([59,  5, 52,  3])
prb_bowtie_LL[1] = np.array([60, 10, 54, 53])
prb_bowtie_LL[2] = np.array([58, 12,  4, 13])
prb_bowtie_LL[3] = np.array([11, 57,  2, 55])

prb_bowtie_LL[4] = np.array([15,  8,  1, 51])
prb_bowtie_LL[5] = np.array([63, 48, 50, 18])
prb_bowtie_LL[6] = np.array([47,  0, 34, 31])
prb_bowtie_LL[7] = np.array([61,  9, 25, 33])
prb_bowtie_LL[8] = np.array([14, 62, 22, 17])

prb_bowtie_LL[9]  = np.array([39, 19, 37, 21])
prb_bowtie_LL[10] = np.array([38, 20, 35, 36])
prb_bowtie_LL[11] = np.array([45, 44, 26, 29])
prb_bowtie_LL[12] = np.array([24,  7, 40, 27])
prb_bowtie_LL[13] = np.array([32, 16, 42, 41])
prb_bowtie_LL[14] = np.array([23, 46, 28, 43])

for prb_id_LL in range(15):
    prb_bowtie_LL[prb_id_LL+15] = prb_bowtie_LL[prb_id_LL] + 64


if __name__ == '__main__':
    prb = prb_bowtie_L
    prb_view = probe_view()
    prb_view.set_data(prb)
    prb_view.run()