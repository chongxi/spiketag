'''
This test case is the basic write and read function for 
`bram_thres` module in FPGA
`bram_thres` module serve as memory interface between host and FPGA for 
1. `threshold`: threshold for spike detection      | 0   -- 255
2. `offset`   : offset DC value for recording      | 256 -- 511
3. `ch_hash`  : channel mapping (ch->ch_hash)      | 512 -- 767
4. `ch_gp_out`: channel groupNO (ch->ch_gp_out)    | 768 -- 1023
'''

from spiketag.fpga.memory_api import *
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    
    x = []
    y = list(range(1024))

    print 'write and read from 0-1023: '

    for i in range(1024):
        write_thr_32(i,i)
        
    for i in range(1024):
        x.append(read_thr_32(i))
    
    if x==y:
        print 'test pass'
    else:
        print 'test fail'
        plt.plot(x, '.')
        plt.plot(y, '.')
        plt.show()
