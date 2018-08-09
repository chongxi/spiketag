from memory_api import *
import numpy as np
import struct

# thres_arr = np.array([-82.62956646, -76.42605996, -82.54043432, -75.53700079,
#                       -76.63554311, -73.02410785, -74.65043008, -72.00293403,
#                       -75.13929951, -74.26652845, -79.01632141, -74.67825559,
#                       -80.10083786, -78.06572939, -84.33438832, -80.27638746])

# thres_arr = np.array([-100.14724392]*32)

class chgpNo(object):
    """
    Must configure for FPGA tranformation to report the groupNo when spike is found
    and transformed
    """
    def __init__(self, nCh=32, base_address=768):
        self.nCh  = nCh
        self.base = base_address
        self.chgpNo = np.zeros(nCh)

    def enable(self, flag):
        if flag is True:
            write_mem_16(self.enable_reg_addres,0b0001)
        elif flag is False:
            write_mem_16(self.enable_reg_addres,0b0000)

    def __setitem__(self, chNo, chgpNo):
        self.chgpNo[chNo] = chgpNo
        if type(chNo) is slice:
            for i,v in enumerate(chgpNo):
                write_thr_32(i+self.base, chgpNo[i], dtype='<I4', binpoint=0) 
        else:
            write_thr_32(chNo+self.base, chgpNo, dtype='<I4', binpoint=0) 

    def __getitem__(self, chNo):
        ch = chNo+self.base
        return read_thr_32(ch, dtype='<I4', binpoint=0)

    def __repr__(self):
        for ch in np.arange(self.nCh):
            print('chgpNo of ch{0} is {1}'.format(ch, self.chgpNo[ch]))
        return 'chgpNo done'


class offset(object):
    """
    """
    def __init__(self, nCh=32, base_address=512):
        self.nCh  = nCh
        self.base = base_address
        self.offset = np.zeros(nCh)

    def enable(self, flag):
        if flag is True:
            write_mem_16(self.enable_reg_addres,0b0001)
        elif flag is False:
            write_mem_16(self.enable_reg_addres,0b0000)

    def __setitem__(self, chNo, offset):
        self.offset[chNo] = offset
        if type(chNo) is slice:
            for i,v in enumerate(offset):
                write_thr_32(i+self.base, offset[i], dtype='<i4', binpoint=13) 
        else:
            write_thr_32(chNo+self.base, offset, dtype='<i4', binpoint=13) 

    def __getitem__(self, chNo):
        ch = chNo+self.base
        return read_thr_32(ch, dtype='<i4', binpoint=13)

    def __repr__(self):
        for ch in np.arange(self.nCh):
            print('offset of ch{0} is {1}'.format(ch, self.offset[ch]))
        return 'offset done'






class channel_hash(object):
    """
    ch ==> hash_code ==> channel No around ch
    this hash_code is stored in ch_unigroup[ch]
    In this class: the ch_unigroup[ch] is write and read intuitively as python convention
    In FPGA:       the ch_unigroup[ch] is organized as a memory structure in bram_thres module in Xike
    Currently, the base address shift is 128, because the first 128 slot is used by threshold
    """
    
    def __init__(self, nCh=32, base_address=256):
        self.nCh  = nCh
        self.base = base_address
        self.ch_unigroup = np.zeros(nCh)

    def __setitem__(self, chNo, ch_group):
        ch_nn0, ch_nn1, ch_nn2, ch_nn3 = ch_group
        # print ch_nn0, ch_nn1, ch_nn2, ch_nn3
        x = struct.unpack('<I', struct.pack('4B', 
                                            ch_nn0, ch_nn1, ch_nn2, ch_nn3))[0]
        ch = chNo+self.base
        write_thr_32(ch, x, dtype='<I4', binpoint=0) 

    def __getitem__(self, chNo):
        ch = chNo+self.base
        x = read_thr_32(ch, dtype='<I', binpoint=0)
        return struct.unpack('4B', struct.pack('<I', x))

    def __str__(self):
        self.hash_repr = ''
        for i in range(self.nCh):
            self.hash_repr += '{}:{}\n'.format(i, self.__getitem__(i))
        return self.hash_repr

    __repr__ = __str__



class threshold(object):
    """
    Class for setting threshold for each channel (up to 1024)
    Connect to a Block RAM(1024x32) in FPGA
    Based on `write_thr_32` and `read_thr_32`
    with virtual device `\\\\.\\xillybus_mem_32`
    The default is 14 bits refractional part

    example:
    thres = threshold(nCh=32)
    thres.enable(True)
    # 1. write
    thres[0] = -70
    thres[1] = -60
    # 2. read
    print thres[0],thres[1]
    # 3. batch write
    thres_arr = np.array([3.1,2,4,6.5])
    thres[:] = thres_arr
    """
    def __init__(self, nCh=32):
        self.enable_reg_addres = 0
        self.nCh = nCh
        # thr_reset(nCh)
        self.enable(True)
        self.thres = np.zeros(nCh)

    def enable(self, flag):
        if flag is True:
            write_mem_16(self.enable_reg_addres,0b0001)
        elif flag is False:
            write_mem_16(self.enable_reg_addres,0b0000)

    def __setitem__(self, chNo, thr):
        self.thres[chNo] = thr 
        if type(chNo) is slice:
            if chNo == slice(None,None,None): 
                for ch in np.arange(self.nCh):
                    if np.size(thr) == 1:
                        write_thr_32(ch, thr,     dtype='<i4', binpoint=13)
                    else:
                        write_thr_32(ch, thr[ch], dtype='<i4', binpoint=13)
            else:
                for i, ch in enumerate(np.arange(chNo.start, chNo.stop, chNo.step)):
                    if np.size(thr) == 1:
                        write_thr_32(ch, thr,     dtype='<i4', binpoint=13)
                    else:
                        write_thr_32(ch, thr[i], dtype='<i4', binpoint=13) 

        else:
            write_thr_32(chNo, thr, dtype='<i4', binpoint=13) 

    def __getitem__(self, chNo):
        return read_thr_32(chNo, dtype='<i4', binpoint=13) 

    def __repr__(self):
        for ch in np.arange(self.nCh):
            print('threshold of ch{0} is {1}'.format(ch, self[ch]))
        return 'threshold enable status: {0}'.format(bool(read_mem_16(self.enable_reg_addres)))

