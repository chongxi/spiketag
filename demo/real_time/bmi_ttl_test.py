from spiketag.command import main
from spiketag.res.GUI.BMI_RASTER_GUI import BMI_RASTER_GUI
import sys
import click
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication
from spiketag.realtime import BMI
import time
import io
import serial
import struct
import datetime as dt
import os

TTLport = '/dev/ttyACM0'
# TTLport = None

class bmi_stream(object):
    """docstring for bmi_stream"""
    def __init__(self, buf):
        super(bmi_stream, self).__init__()
        self.buf = buf
        self.output = struct.unpack('<7i', self.buf)        
        self.timestamp, self.grp_id, self.fet0, self.fet1, self.fet2, self.fet3, self.spk_id = self.output


## python main
if __name__ == '__main__':
    app = QApplication(sys.argv) 
    gui = BMI_RASTER_GUI(fet_file='./fet.bin', t_window=10e-3, view_window=10, ttlport=TTLport)
    # gui.bmi.set_binner(bin_size = 40e-3, B_bins=7) # 40ms bin size, 7 bins
    gui.bmi.dec = None
    # from spiketag.analysis import load_decoder
    # dec = load_decoder('/home/chongxi/disk1/temp/dec')
    # score = dec.score(smooth_sec=2)
    # gui.bmi.set_decoder(dec, dec_file='./_dec', score=False)
    # print(gui.bmi.dec.fields.shape[0], gui.bmi.dec.t_step, gui.bmi.dec.t_window)
    # print(score)

    # @gui.bmi.binner.connect
    # def on_decode(X):
    #     y, post_2d = gui.bmi.dec.predict_rt(X)
    #     post_2d /= post_2d.sum()
    #     max_post = post_2d.max()
    #     gui.bmi.TTLserial.write(b'a')  # send TTL out

        

    gui.show()
    sys.exit(app.exec_())

    # bmi = BMI(prb=None, fetfile='./fet.bin', ttlport=TTLport)
    # bmi.start()
    # time.sleep(20)
    # bmi.stop()


    # TTLserial = serial.Serial(port=TTLport, baudrate=115200, timeout=0)
    # r32 = io.open('/dev/xillybus_fet_clf_32', 'rb')
    # size = 7*4

    # while True:
    #     try:
    #         tic = dt.datetime.now()
    #         buf = r32.read(size)
    #         bmi_output = bmi_stream(buf)
    #         if bmi_output.grp_id == 39:
    #             # print(bmi_output.timestamp, bmi_output.grp_id, bmi_output.spk_id)
    #             if TTLserial is not None:
    #                 TTLserial.write(b'a')
    #             toc = dt.datetime.now()
    #             delta = toc.microsecond - tic.microsecond
    #             print(delta/1e3) # ms
                
    #     except KeyboardInterrupt:
    #         break
    
    # r32.close()
    # TTLserial.close()
    # print('done')
