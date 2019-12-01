from spiketag.fpga.memory_api import *
import numpy as np
import struct

class pca_hash(object):
    """
    address: 128 - 1952(128+76*32)
    ch ==> hash_code ==> pca coef around ch
    this hash_code is stored in pca[ch]
    In this class: the pca[ch] is write and read intuitively as python convention
    In FPGA:       the pca[ch] is organized as a memory structure in bram_thres module in Xike
    Currently, the base address shift is 128, because the first 128 slots are used by shift and scale

    pca[ch] is a (ndim,4) matrix

    ----------------- * oooo 
         x:(ndim)       |||| (4*8bits)
                        ||||
                        ||||
                        ||||
                        ||||
                        ||||
                        ||||
                        ||||
                        ||||

    y = a(xP+b)
    P is this pca[ch], it converts (ndim,) spike waveform to (4,)

    In FPGA side, pca[ch] is ndim 32 bits, each 32 bits are composed by 4*8bits:(pca0,pca1,pca2,pca3)
    write_pca_in and read_pca_out function write and read one 32 bits into FPGA memory
    """
    def __init__(self, nCh=32, base_address=128, ndim=76):
        self.nCh  = nCh
        self.base = base_address
        self.pca = np.zeros(nCh)
        self.dim = ndim 

    def write_pca_in(self, i, pca_in):
        # print i, pca_in
        pca_in = np.floor(np.asarray(pca_in)*2**7)
        pca_in = pca_in.astype(np.int32)
        pca0, pca1, pca2, pca3 = pca_in
        x = struct.unpack('<i', struct.pack('4b', 
                                            pca0, pca1, pca2, pca3))[0]
        write_tat_32(i, x, dtype='<i', binpoint=0) 

    def __setitem__(self, grpNo, pca_comp):
        ch = grpNo*self.dim + self.base
        for i, pca_in in enumerate(pca_comp):
            self.write_pca_in(ch+i, pca_in)
        
    def read_pca_out(self, i):
        x = read_tat_32(i, dtype='<i', binpoint=0)
        y = struct.unpack('bbbb', struct.pack('<i', int(x)))
        y = np.asarray(y).astype(np.float32)
        y = y/(2**7)
        return y

    def __getitem__(self, grpNo):
        pca_comp = np.zeros((self.dim, 4))
        ch = grpNo*self.dim + self.base
        for i in range(self.dim):
            pca_comp[i] = self.read_pca_out(ch+i)
        return pca_comp

    def get_hex(self, grpNo):
        pca_comp_hex = []
        ch = grpNo*self.dim + self.base
        for i in range(self.dim):
            addr = (ch+i) * 4
            r32 = open('/dev/xillybus_template_32', 'rb')
            r32.seek(addr)
            hexstring = r32.read(4)
            pca_comp_hex.append(hexstring)
            r32.close()     
        return pca_comp_hex
         

class shift_hash(object):
    """
    32-127
    """
    def __init__(self, nCh=32, base_address=32):
        self.nCh = nCh
        self.base = base_address
        self.dim   = 4
        # for i in range(self.nCh):
        #     self.__setitem__(i,np.zeros(self.dim,))
        
    def __setitem__(self, chNo, shift):
        ch = chNo*self.dim + self.base
        for i, _shift in enumerate(shift):
            write_tat_32(ch+i, _shift, dtype='<i', binpoint=13) 

    def __getitem__(self, chNo):
        ch = chNo*self.dim + self.base
        x = []
        for i in range(self.dim):
            x.append(read_tat_32(ch+i, dtype='<i', binpoint=13))
        return np.array(x)

    def __repr__(self):
        _shift = np.zeros((self.nCh, self.dim))
        for i in range(self.nCh):
            _shift[i] = self.__getitem__(i)
        print(_shift)
        return ' '

class scale_hash(object):
    """
    0-31
    """
    def __init__(self, nCh=32, base_address=0):
        self.nCh = nCh
        self.base = base_address
        self.dim = 1
        # for i in range(self.nCh):
        #     self.__setitem__(i,0)

    def __setitem__(self, chNo, _scale):
        ch = chNo + self.base
        write_tat_32(ch, _scale, dtype='<i', binpoint=13) 

    def __getitem__(self, chNo):
        ch = chNo + self.base
        x0 = read_tat_32(ch, dtype='<i', binpoint=13) 
        return x0

    def __repr__(self):
        _scale = np.zeros((self.nCh, self.dim))
        for i in range(self.nCh):
            _scale[i] = self.__getitem__(i)
        print(_scale)
        return ' '


class vq_hash(object):
    """
    ch ==> hash_code ==> vq in ch
    this hash_code is stored in vq[ch]
    In this class: the vq[ch] is write and read intuitively as python convention
    In FPGA:       the vq[ch] is organized as a memory structure in bram_thres module in Xike
    Currently, the base address shift is 1952 = 128 + 57*32
    """
    def __init__(self, nCh=32, base_address=1952, ndim=500):
        self.nCh  = nCh
        self.base = base_address
        self.vq   = np.zeros(nCh)
        self.dim  = ndim

    def write_vq_in(self, i, vq_in):
        vq_in = np.floor(np.asarray(vq_in)*2**7)
        vq_in = vq_in.astype(np.int32)
        vq0, vq1, vq2, vq3 = vq_in
        x = struct.unpack('<i', struct.pack('bbbb', 
                                            vq0, vq1, vq2, vq3))[0]
        ch = i + self.base
        write_tat_32(ch, x, dtype='<i', binpoint=0) 

    def read_vq_out(self, i):
        ch = i + self.base
        x = read_tat_32(ch, dtype='<i', binpoint=0)
        y = struct.unpack('bbbb', struct.pack('<i', int(x)))
        y = np.asarray(y).astype(np.float32)
        y = y/(2**7)
        return y

    def __setitem__(self, grpNo, vq):
        ch = grpNo*self.dim
        for i, vq_in in enumerate(vq):
            self.write_vq_in(ch+i, vq_in)
 
    def __getitem__(self, grpNo):
        vq = np.zeros((self.dim, 4))
        ch = grpNo*self.dim
        for i in range(self.dim):
            vq[i] = self.read_vq_out(ch+i)
        return vq


class label_hash(object):
    """
    grp_id ==> hash_code ==> labels in the group
    this hash_code is stored in label[ch]
    In this class: the label[ch] is write and read intuitively as python convention
    In FPGA:       the label[ch] is organized as a memory structure in bram_thres module in Xike
    Currently, the base address shift is 1952 = 128 + 57*32
    """
    def __init__(self, nCh=32, base_address=1952, ndim=500):
        self.nCh  = nCh
        self.base = base_address
        self.lb   = np.zeros(nCh)
        self.dim  = ndim

    def write_lb_in(self, i, lb_in):
        lb_in = lb_in.astype(np.int32)
        ch = i + self.base
        write_tat_32(ch, lb_in, dtype='<i', binpoint=0) 

    def read_lb_out(self, i):
        ch = i + self.base
        lb = read_tat_32(ch, dtype='<i', binpoint=0)
        return lb

    def __setitem__(self, grpNo, lb):
        ch = grpNo*self.dim
        for i, lb_in in enumerate(lb):
            self.write_lb_in(ch+i, lb_in)
 
    def __getitem__(self, grpNo):
        lb = np.zeros((self.dim,)).astype(np.int32)
        ch = grpNo*self.dim
        for i in range(self.dim):
            lb[i] = self.read_lb_out(ch+i)
        return lb

    def __repr__(self):
        # self._labels = np.zeros((self.nCh, self.dim))
        self._labels = ''
        for i in range(self.nCh):
            self._labels += ('group {} labels: {}\n'.format(i, list(np.unique(self.__getitem__(i)))))
        return self._labels
