import io
import os
import time
import sys
import struct
import socket
import numpy as np
import torch as torch
from collections import namedtuple
from spiketag.fpga import FPGA
from torch.multiprocessing import Process, Pipe, SimpleQueue 
from ..utils.utils import EventEmitter, Timer
from ..realtime import Binner 


class bmi_stream(object):
    """
    data structure used to read from FPGA bmi data stream
    """
    def __init__(self, buf):
        super(bmi_stream, self).__init__()
        self.buf = buf
        self.output = struct.unpack('<8i', self.buf)        
        self.timestamp, self.grp_id, self.fet0, self.fet1, self.fet2, self.fet3, self.spk_id, self.spk_energy = self.output


class bmi_packet(object):
    '''
    Data structure to read single bmi_packet from spike_df (pandas dataframe)
    This class is used to simulate a bmi recording + decoding process, by replaying bmi packets from `a recorded spike data frame` to a Binner, which was used in real-time 
    by connecting to a `on_decode` method. 

    Example:
        # ! 1. prepare spike_df
        spkdf = pc.spike_df[(pc.spike_df.frame_id<=9) & (pc.spike_df.frame_id>0)]
        bmi_data = bmi_packet(spkdf)
        # ! 2. prepare a binner
        binner = Binner(bin_size=0.1, n_id=scv_full.shape[1], n_bin=7)
        # ! 3. prepare the on_decode method for binner
        rt_scv = []
        @binner.connect
        def on_decode(X):
            rt_scv.append(X)
        # ! 4. simulate a bmi recording process
        for bmi_output in bmi_data:
            binner.input(bmi_output)
    '''
    def __init__(self, spike_df, fs=25000.):
        self.timestamp = 0
        self.spk_id = 0
        self.fs = fs
        self.spike_df = spike_df
        self.bmi_output = namedtuple('bmi_output', ['timestamp', 'spk_id'])
    
    def __len__(self):
        return len(self.spike_df)

    def __getitem__(self, i):
        spike = self.spike_df.iloc[i]
        self.timestamp = spike.frame_id * self.fs
        self.spk_id = spike.spike_id
        return self.bmi_output(self.timestamp, self.spk_id)
    
    def __repr__(self) -> str:
        return f'a bmi_packet containing {len(self.spike_df)} single-spike packets'

class BMI(object):
    """
    BMI: https://github.com/chongxi/spiketag/issues/58
    1. receive bmi output from FPGA through a pcie channel, save to a file
    2. parse the bmi output (timestamp, group_id, fet[:4], spike_id)
    3. send the output to the binner, which will emit event to trigger decoder each time a new bin is completely filled
    4. put the output into the queue for gui to display

    A) configure mode:
    >>> bmi = BMI(prb)
    in this case, `bmi.fpga` is used to configure FPGA model parameters

    B) real-time spike inference mode:
    >>> bmi = BMI(prb, fetfile)
    in this case, not only `bmi.fpga` can be used to configure FPGA, but also these parameters 
    should be read out to configure higher-level containers such as a BMI GUI
   
    C) real-time spike inference mode:
    >>> bmi = BMI(prb, fetfile, ttlport)
    We can set the TTLport (a serial port, e.g. /dev/ttyACM0) to allow the PC to generate a TTL signal with customized trigger

    D) Additional to the spike inference, the inferred spikes can be fed into `binner` and then to a decoder
    >>> bmi.set_binner(bin_size, B_bins) 
    >>> bmi.set_decoder(dec, dec_file='dec')

    E) Start bmi with or without a `gui_queue` for visualization
    >>> bmi.start(gui_queue=True)
    >>> bmi.stop() 

    F) Read out the content in the bmi.gui_queue for visualization
    >>> bmi.gui_queue.get()
    """

    def __init__(self, prb=None, fetfile=None, ttlport=None):
        if prb is not None:
            self.prb = prb
            self.ngrp = prb.n_group
            # self.group_idx = np.array(list(self.prb.grp_dict.keys()))
            self.fpga = FPGA(self.prb)
        else:
            self.ngrp = 40   # by default
            self.fpga = FPGA()   # by default
        print('{} groups on probe'.format(self.ngrp))
        print('{} groups is configured in the FPGA: {}'.format(len(self.fpga.configured_groups), 
                                                               self.fpga.configured_groups))
        print('{} neurons are configured in the FPGA'.format(self.fpga.n_units+1))
        print('---1. BMI spike-model initiation succeed---\n')

        if fetfile is not None:
            self.init_bmi_packet_channel()
            self.fetfile = fetfile
            self.fd = os.open(self.fetfile, os.O_CREAT | os.O_WRONLY | os.O_NONBLOCK)
            print('spike-id and feature is saved to {}\n'.format(self.fetfile)) 

        if ttlport is not None:
            import serial
            self.TTLserial = serial.Serial(port=ttlport, baudrate=115200, timeout=0)
            self.TTLserial.flushInput()
            self.TTLserial.flushOutput()
        else:
            self.TTLserial = None

        self.binner = None

    def close(self):
        self.r32.close()

    def init_bmi_packet_channel(self):
        self.r32 = io.open('/dev/xillybus_fet_clf_32', 'rb', buffering=4)  # this buffer size is critical for performance
        self._size = 8*4  # 8 samples, 4 bytes/sample
        self.bmi_buf = None
        print('spike-id packet channel is opened\n')

    def set_binner(self, bin_size, B_bins):
        '''
        set bin size, N neurons and B bins for the binner
        '''
        N_units = self.fpga.n_units + 1 # The unit #0, no matter from which group, is always noise
        self.binner = Binner(bin_size, N_units, B_bins)    # binner initialization (space and time)      
        print('BMI binner: {} bins {} units, each bin is {} seconds'.format(B_bins, N_units, bin_size))  
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

    def set_decoder(self, dec, dec_file=None):
        self.dec = dec
        if dec_file is not None:
            self.dec.save(dec_file)
        print('---3. BMI Decoder initiation succeed---\n')
        

    def read_bmi(self):
        '''
        take buf from pcie channel '/dev/xillybus_fet_clf_32'
        filter the output with defined rules according to timestamp and grp_id
        each bmi_output is a compressed spike: 
        (timestamp, grp_id, fet0, fet1, fet2, fet3, spk_id)
        '''
        # filled = False
        # while not filled:
        self.buf = self.r32.read(self._size)
        os.write(self.fd, self.buf)
        # bmi_output = struct.unpack('<7i', self.buf)
        bmi_output = bmi_stream(self.buf)
        # bmi filter
        # if bmi_output.spk_id > 0:
            # filled=True

        ###############################################################################
        ##### Customized code for testing (comment out if not in test mode) ###########
        # if bmi_output.grp_id == 39:
        #     if self.TTLserial is not None and self.dec is not None:
        #         # decoding
        #         # y, post_2d = self.dec.predict_rt(self.binner.output)
        #         # post_2d /= post_2d.sum()
        #         # max_post = post_2d.max()

        #         # output TTL
        #         self.TTLserial.write(b'a')
        #         self.TTLserial.flush()
        ###############################################################################
        ###############################################################################

        return bmi_output


    def BMI_core_func(self, gui_queue, model=None):
        '''
        A daemon process dedicated on reading data from PCIE and update
        the shared memory with other processors: shared_arr 

        This process func starts when self.start()
                          it ends with self.stop()
        '''
        # os.nice(-20) # makes this process almost real-time priority
        self.model = model
        
        while True:
            with Timer('real-time decoding', verbose=False):
                bmi_output = self.read_bmi()
                # timestamp, grp_id, fet0, fet1, fet2, fet3, spk_id = bmi_output 
                # ----- real-time processing the BMI output ------
                # ----- This section should cost < 100us -----
                
                ##### real-time decoder
                # 1. binner
                # print(bmi_output.timestamp, bmi_output.grp_id)
                if self.binner is not None:
                    self.binner.input(bmi_output) 
                # print(bmi_output.output)
                # 2. gui queue (optional)
                ##### queue for visualization on GUI
                if self.gui_queue is not None:
                    self.gui_queue.put(bmi_output.output)

                ##### file for visualization

                # ----- This section should cost < 100us -----


    def start(self, gui_queue=False, model=None):
        if gui_queue:
            self.gui_queue = SimpleQueue()
        else:
            self.gui_queue = None
        self.fpga_process = Process(target=self.BMI_core_func, name='fpga', args=(self.gui_queue, model))
        self.fpga_process.daemon = True
        self.fpga_process.start()  


    def stop(self):
        self.fpga_process.terminate()
        self.fpga_process.join()
        self.gui_queue = None

