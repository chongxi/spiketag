import io
import os
import time
import sys
import struct
import socket
import numpy as np
import torch as torch
from spiketag.fpga import xike_config
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


class BMI(object):
    """
    BMI 
    1. receive bmi output from FPGA through a pcie channel, save to a file
    2. parse the bmi output, filter the bmi output
    3. send the output to the decoder
    4. put the output into the queue for gui to display
    """
    def __init__(self, prb, fetfile='./fet.bin'):
        self.prb = prb
        self.ngrp = prb.n_group
        self.group_idx = np.array(list(self.prb.grp_dict.keys()))
        self.fetfile = fetfile
        self.init()

    def close(self):
        self.r32.close()

    def init(self):
        self.r32 = io.open('/dev/xillybus_fet_clf_32', 'rb')
        # self.r32_buf = io.BufferedReader(r32)
        self.fd = os.open(self.fetfile, os.O_CREAT | os.O_WRONLY | os.O_NONBLOCK)
        self._size = 7*4  # 7 samples, 4 bytes/sample
        self.bmi_buf = None

        self.fpga = xike_config(self.prb)
        print('{} groups on probe'.format(self.ngrp))
        print('{} groups is configured in the FPGA: {}'.format(len(self.fpga.configured_groups), 
                                                               self.fpga.configured_groups))
        print('{} neurons are configured in the FPGA'.format(self.fpga.n_units+1))
        print('---1. BMI spike-model initiation succeed---\n')


    def set_binner(self, bin_size, B_bins):
        '''
        set bin size, N neurons and B bins for the binner
        '''
        N_units = self.fpga.n_units + 1
        self.binner = Binner(bin_size, N_units, B_bins)    # binner initialization (space and time)      
        print('BMI binner: {} bins {} units, each bin is {} seconds'.format(N_units, B_bins, bin_size))  
        print('---2. BMI binner initiation succeed---\n')
        # @self.binner.connect
        # def on_decode(X):
        #     # print(self.binner.nbins, np.sum(self.binner.output), self.binner.count_vec.shape)
        #     print(self.binner.nbins, self.binner.count_vec.shape, X.shape, np.sum(X))
     
    # def shared_mem_init(self):
    #     n_spike_count_vector = len(self.prb.grp_dict.keys())
    #     # trigger task using frame counter
    #     self.spike_count_vector = torch.zeros(n_spike_count_vector,)
    #     self.spike_count_vector.share_memory_()

    def set_decoder(self, dec, dec_result_file=None):
        print('Training decoder for the bmi')
        self.dec = dec
        self.dec.resample(t_step=self.binner.bin_size, t_window=self.binner.bin_size*self.binner.B)
        self.dec.partition(training_range=[0.0, 1.0], valid_range=[0.5, 0.6], testing_range=[0.0, 1.0])
        score = self.dec.auto_pipeline(smooth_sec=2) # 2 seconds smooth for scoring

        if dec_result_file is not None:
           self.dec_result =  os.open(dec_result_file, os.O_CREAT | os.O_WRONLY | os.O_NONBLOCK)

        print('connecting decoder to the bmi for real-time control')
        @self.binner.connect
        def on_decode(X):
            # print(self.binner.nbins, self.binner.count_vec.shape, X.shape, np.sum(X))
            if dec.name == 'NaiveBayes':
                X = np.sum(X, axis=0)
            y = self.dec.predict(X)
            print('pos:{0}, time:{1:.5f}'.format(y, self.binner.current_time))
            os.write(self.dec_result, y)
        print('---3. BMI Decoder initiation succeed---\n')
        

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
            if bmi_output.spk_id > 0:
                filled=True
                return bmi_output


    def BMI_core_func(self, gui_queue):
        '''
        A daemon process dedicated on reading data from PCIE and update
        the shared memory with other processors: shared_arr 

        This process func starts when self.start()
                          it ends with self.stop()
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
                self.binner.input(bmi_output) 
                # print(bmi_output.output)
                # 2. gui queue (optional)
                ##### queue for visualization on GUI
                if self.gui_queue is not None:
                    self.gui_queue.put(bmi_output.output)

                ##### file for visualization

                # ----- This section should cost < 100us -----


    def start(self, gui_queue=False):
        if not self.binner:
            print('set binner first')
        if gui_queue:
            self.gui_queue = SimpleQueue()
        else:
            self.gui_queue = None
        self.fpga_process = Process(target=self.BMI_core_func, name='fpga', args=(self.gui_queue,)) #, args=(self.pipe_jovian_side,)
        self.fpga_process.daemon = True
        self.fpga_process.start()  


    def stop(self):
        self.fpga_process.terminate()
        self.fpga_process.join()
        self.gui_queue = None

