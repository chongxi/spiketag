import io
import os
import time
import sys
import struct
import socket
import numpy as np
import torch as torch
from torch.multiprocessing import Process, Pipe, SimpleQueue 
from ..utils.utils import EventEmitter, Timer
from ..realtime import Binner 


class bmi_stream(object):
    """docstring for bmi_stream"""
    def __init__(self, buf):
        super(bmi_stream, self).__init__()
        self.buf = buf
        self.output = struct.unpack('<7i', self.buf)        
        self.timestamp, self.grp_id, self.fet0, self.fet1, self.fet2, self.fet3, self.spk_id = self.output


class bmi_recv(object):
    """
    bmi_recv 
    1. receive bmi output from FPGA through a pcie channel, save to a file
    2. parse the bmi output, filter the bmi output
    3. send the output to the decoder
    4. put the output into the queue for gui to display
    """
    def __init__(self, prb, fetfile):
        self.prb = prb
        self.ngrp = prb.n_group
        self.group_idx = np.array(list(self.prb.grp_dict.keys()))
        print('{} groups on probe: {}'.format(self.ngrp, self.group_idx))
        self.fetfile = fetfile
        self.init()

    def close(self):
        self.r32.close()

    def init(self):
        self.r32 = io.open('/dev/xillybus_fet_clf_32', 'rb')
        # self.r32_buf = io.BufferedReader(r32)
        self.fd = os.open(self.fetfile, os.O_CREAT | os.O_WRONLY | os.O_NONBLOCK)
        self._size = 7*4  # 6 samples, 4 bytes/sample
        self.bmi_buf = None


    def set_binner(self, bin_size, n_id, n_bin):
        self.binner = Binner(bin_size, n_id, n_bin)    # binner initialization (space and time)
        @self.binner.connect
        def on_decode():
            print(self.binner.nbins, np.sum(self.binner.output), self.binner.count_vec.shape)        

    # def shared_mem_init(self):
    #     n_spike_count_vector = len(self.prb.grp_dict.keys())
    #     # trigger task using frame counter
    #     self.spike_count_vector = torch.zeros(n_spike_count_vector,)
    #     self.spike_count_vector.share_memory_()


    def read_bmi(self):
        '''
        take buf from pcie channel '/dev/xillybus_fet_clf_32'
        filter the output with defined rules according to timestamp and grp_id
        each bmi_output is a compressed spike: 
        (timestamp, grp_id, fet0, fet1, fet2, fet3, spk_id)
        '''
        filled = False
        while not filled:
            self.buf = self.r32.read(self._size)
            os.write(self.fd, self.buf)
            # bmi_output = struct.unpack('<7i', self.buf)
            bmi_output = bmi_stream(self.buf)
            # bmi filter
            if (bmi_output.timestamp > 500) and (bmi_output.grp_id in self.group_idx):
                filled=True
                return bmi_output

    def BMI_core_func(self, gui_queue):
        '''
        A daemon process dedicated on reading data from PCIE and update
        the shared memory with other processors: shared_arr 
        '''
        
        while True:
            with Timer('real-time decoding', verbose=False):
                bmi_output = self.read_bmi()
                # timestamp, grp_id, fet0, fet1, fet2, fet3, spk_id = bmi_output 
                # ----- real-time processing the BMI output ------
                # ----- This section should cost < 100us -----
                    
                ##### real-time decoder
                # 1. binner
                # print(bmi_output.timestamp, bmi_output.grp_id)
                self.binner.input(bmi_output, type='individual_spike') 
                # print(bmi_output)
                # nbin = timestamp//self.time 
                ##### queue for visualization on GUI
                gui_queue.put(bmi_output.output)

                ##### file for visualization

                # ----- This section should cost < 100us -----


    def start(self):
        self.gui_queue = SimpleQueue()
        self.fpga_process = Process(target=self.BMI_core_func, name='fpga', args=(self.gui_queue,)) #, args=(self.pipe_jovian_side,)
        self.fpga_process.daemon = True
        self.fpga_process.start()  


    def stop(self):
        self.fpga_process.terminate()
        self.fpga_process.join()

