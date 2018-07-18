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

#####################################
# neuronexus
#####################################
prb_neuronexus = probe(shank_no = 20, prb_type = 'neuronexus')
prb_neuronexus.shanks[0].l=[41, 38, 40, 39]
prb_neuronexus.shanks[0].r=[37, 42, 36, 43]
prb_neuronexus.shanks[0].xl=0
prb_neuronexus.shanks[0].yl=3
prb_neuronexus.shanks[0].xr=3
prb_neuronexus.shanks[0].yr=0
prb_neuronexus.shanks[1].l=[45, 34, 44, 35]
prb_neuronexus.shanks[1].r=[33, 46, 32, 47]
prb_neuronexus.shanks[1].xl=40
prb_neuronexus.shanks[1].yl=3
prb_neuronexus.shanks[1].xr=43
prb_neuronexus.shanks[1].yr=0
prb_neuronexus.shanks[2].l=[18, 17, 31, 16]
prb_neuronexus.shanks[2].r=[29, 19, 20, 27]
prb_neuronexus.shanks[2].xl=80
prb_neuronexus.shanks[2].yl=3
prb_neuronexus.shanks[2].xr=83
prb_neuronexus.shanks[2].yr=0
prb_neuronexus.shanks[3].l=[23, 25, 22, 21]
prb_neuronexus.shanks[3].r=[24, 26, 28, 30]
prb_neuronexus.shanks[3].xl=120
prb_neuronexus.shanks[3].yl=3
prb_neuronexus.shanks[3].xr=123
prb_neuronexus.shanks[3].yr=0
prb_neuronexus.shanks[4].l=[7, 5, 3, 1]
prb_neuronexus.shanks[4].r=[ 8,  6,  9, 10]
prb_neuronexus.shanks[4].xl=160
prb_neuronexus.shanks[4].yl=3
prb_neuronexus.shanks[4].xr=163
prb_neuronexus.shanks[4].yr=0
prb_neuronexus.shanks[5].l=[ 2, 12, 11,  4]
prb_neuronexus.shanks[5].r=[13, 14,  0, 15]
prb_neuronexus.shanks[5].xl=200
prb_neuronexus.shanks[5].yl=3
prb_neuronexus.shanks[5].xr=203
prb_neuronexus.shanks[5].yr=0
prb_neuronexus.shanks[6].l=[62, 49, 63, 48]
prb_neuronexus.shanks[6].r=[50, 61, 51, 60]
prb_neuronexus.shanks[6].xl=240
prb_neuronexus.shanks[6].yl=3
prb_neuronexus.shanks[6].xr=243
prb_neuronexus.shanks[6].yr=0
prb_neuronexus.shanks[7].l=[58, 53, 59, 52]
prb_neuronexus.shanks[7].r=[54, 57, 55, 56]
prb_neuronexus.shanks[7].xl=280
prb_neuronexus.shanks[7].yl=3
prb_neuronexus.shanks[7].xr=283
prb_neuronexus.shanks[7].yr=0
prb_neuronexus.shanks[8].l=[105, 102, 104, 103]
prb_neuronexus.shanks[8].r=[101, 106, 100, 107]
prb_neuronexus.shanks[8].xl=0
prb_neuronexus.shanks[8].yl=93
prb_neuronexus.shanks[8].xr=3
prb_neuronexus.shanks[8].yr=90
prb_neuronexus.shanks[9].l=[109,  98, 108,  99]
prb_neuronexus.shanks[9].r=[ 97, 110,  96, 111]
prb_neuronexus.shanks[9].xl=40
prb_neuronexus.shanks[9].yl=93
prb_neuronexus.shanks[9].xr=43
prb_neuronexus.shanks[9].yr=90
prb_neuronexus.shanks[10].l=[82, 81, 95, 80]
prb_neuronexus.shanks[10].r=[93, 83, 84, 91]
prb_neuronexus.shanks[10].xl=80
prb_neuronexus.shanks[10].yl=93
prb_neuronexus.shanks[10].xr=83
prb_neuronexus.shanks[10].yr=90
prb_neuronexus.shanks[11].l=[87, 89, 86, 85]
prb_neuronexus.shanks[11].r=[88, 90, 92, 94]
prb_neuronexus.shanks[11].xl=120
prb_neuronexus.shanks[11].yl=93
prb_neuronexus.shanks[11].xr=123
prb_neuronexus.shanks[11].yr=90
prb_neuronexus.shanks[12].l=[71, 69, 67, 65]
prb_neuronexus.shanks[12].r=[72, 70, 73, 74]
prb_neuronexus.shanks[12].xl=160
prb_neuronexus.shanks[12].yl=93
prb_neuronexus.shanks[12].xr=163
prb_neuronexus.shanks[12].yr=90
prb_neuronexus.shanks[13].l=[66, 76, 75, 68]
prb_neuronexus.shanks[13].r=[77, 78, 64, 79]
prb_neuronexus.shanks[13].xl=200
prb_neuronexus.shanks[13].yl=93
prb_neuronexus.shanks[13].xr=203
prb_neuronexus.shanks[13].yr=90
prb_neuronexus.shanks[14].l=[126, 113, 127, 112]
prb_neuronexus.shanks[14].r=[114, 125, 115, 124]
prb_neuronexus.shanks[14].xl=240
prb_neuronexus.shanks[14].yl=93
prb_neuronexus.shanks[14].xr=243
prb_neuronexus.shanks[14].yr=90
prb_neuronexus.shanks[15].l=[122, 117, 123, 116]
prb_neuronexus.shanks[15].r=[118, 121, 119, 120]
prb_neuronexus.shanks[15].xl=280
prb_neuronexus.shanks[15].yl=93
prb_neuronexus.shanks[15].xr=283
prb_neuronexus.shanks[15].yr=90
prb_neuronexus.shanks[16].l=[148, 146, 145, 144]
prb_neuronexus.shanks[16].r=[149, 150, 159, 158]
prb_neuronexus.shanks[16].xl=0
prb_neuronexus.shanks[16].yl=183
prb_neuronexus.shanks[16].xr=3
prb_neuronexus.shanks[16].yr=180
prb_neuronexus.shanks[17].l=[153, 154, 155, 157]
prb_neuronexus.shanks[17].r=[152, 156, 151, 147]
prb_neuronexus.shanks[17].xl=40
prb_neuronexus.shanks[17].yl=183
prb_neuronexus.shanks[17].xr=43
prb_neuronexus.shanks[17].yr=180
prb_neuronexus.shanks[18].l=[135, 131, 136, 140]
prb_neuronexus.shanks[18].r=[134, 133, 132, 130]
prb_neuronexus.shanks[18].xl=80
prb_neuronexus.shanks[18].yl=183
prb_neuronexus.shanks[18].xr=83
prb_neuronexus.shanks[18].yr=180
prb_neuronexus.shanks[19].l=[138, 137, 128, 129]
prb_neuronexus.shanks[19].r=[139, 141, 142, 143]
prb_neuronexus.shanks[19].xl=120
prb_neuronexus.shanks[19].yl=183
prb_neuronexus.shanks[19].xr=123
prb_neuronexus.shanks[19].yr=180

prb_neuronexus.auto_pos()
prb_neuronexus[0]= np.array([41, 38, 37, 42])
prb_neuronexus[1]= np.array([40, 39, 36, 43])
prb_neuronexus[2]= np.array([45, 34, 33, 46])
prb_neuronexus[3]= np.array([44, 35, 32, 47])
prb_neuronexus[4]= np.array([18, 17, 29, 19])
prb_neuronexus[5]= np.array([31, 16, 20, 27])
prb_neuronexus[6]= np.array([23, 25, 24, 26])
prb_neuronexus[7]= np.array([22, 21, 28, 30])
prb_neuronexus[8]= np.array([7, 5, 8, 6])
prb_neuronexus[9]= np.array([ 3,  1,  9, 10])
prb_neuronexus[10]= np.array([ 2, 12, 13, 14])
prb_neuronexus[11]= np.array([11,  4,  0, 15])
prb_neuronexus[12]= np.array([62, 49, 50, 61])
prb_neuronexus[13]= np.array([63, 48, 51, 60])
prb_neuronexus[14]= np.array([58, 53, 54, 57])
prb_neuronexus[15]= np.array([59, 52, 55, 56])
prb_neuronexus[16]= np.array([105, 102, 101, 106])
prb_neuronexus[17]= np.array([104, 103, 100, 107])
prb_neuronexus[18]= np.array([109,  98,  97, 110])
prb_neuronexus[19]= np.array([108,  99,  96, 111])
prb_neuronexus[20]= np.array([82, 81, 93, 83])
prb_neuronexus[21]= np.array([95, 80, 84, 91])
prb_neuronexus[22]= np.array([87, 89, 88, 90])
prb_neuronexus[23]= np.array([86, 85, 92, 94])
prb_neuronexus[24]= np.array([71, 69, 72, 70])
prb_neuronexus[25]= np.array([67, 65, 73, 74])
prb_neuronexus[26]= np.array([66, 76, 77, 78])
prb_neuronexus[27]= np.array([75, 68, 64, 79])
prb_neuronexus[28]= np.array([126, 113, 114, 125])
prb_neuronexus[29]= np.array([127, 112, 115, 124])
prb_neuronexus[30]= np.array([122, 117, 118, 121])
prb_neuronexus[31]= np.array([123, 116, 119, 120])
prb_neuronexus[32]= np.array([148, 146, 149, 150])
prb_neuronexus[33]= np.array([145, 144, 159, 158])
prb_neuronexus[34]= np.array([153, 154, 152, 156])
prb_neuronexus[35]= np.array([155, 157, 151, 147])
prb_neuronexus[36]= np.array([135, 131, 134, 133])
prb_neuronexus[37]= np.array([136, 140, 132, 130])
prb_neuronexus[38]= np.array([138, 137, 139, 141])
prb_neuronexus[39]= np.array([128, 129, 142, 143])


if __name__ == '__main__':
    prb = prb_bowtie_L
    prb_view = probe_view()
    prb_view.set_data(prb)
    prb_view.run()
