import numpy as np
import logging
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Pipe, Lock
import time
import os
import io
from spiketag.view import wave_view
from spiketag.utils import Timer
from vispy import app,keys
import torch

info = mp.get_logger().info


# Create two instances of MyRect, each using canvas.scene as their parent
nCh = 32 #32 channel number
# Number of cols and rows in the table.
nrows = 16
ncols = 2
# Number of channels.
m = nrows*ncols
# Number of samples per channel.
npts = 256*2
# Generate the signals as a (nCh, npts) array.
init_data = np.random.randn(npts,nCh).astype(np.float32)
init_data[:,4] = 1 

wview = wave_view(init_data)

@wview.connect
def on_key_press(event):
    if event.key is keys.SPACE:
        if not timer.running:   timer.start()  # reading thread
        else:                   timer.stop()

# shared_arr = multiprocessing.Array(ctypes.c_double, npts*nCh)
shared_arr = torch.from_numpy(init_data)
shared_arr.share_memory_()
lock = Lock()

def update_show(ev):
    '''
    update display, roll the multichannel data according to 
    test1: Generated gaussian data in the same process
    test2: From shared numpy array (look into tonumpyarray)
    ''' 
    # test1 ###########################
    # n = npts
    # d = np.random.randn(n*16,1).reshape(-1,16)
    ###################################################

    # test2 #####################################
    # with shared_arr.get_lock():
        # d = tonumpyarray(shared_arr).astype(np.float32)
    with Timer('update'):
        d = shared_arr.numpy().astype(np.float32)
        m = d.max()
        if m != 0:
            # d = d.reshape(-1, 32)/m
            d /= m
            wview.waves1.set_data(d)

timer = app.Timer(connect=update_show, interval=0)
timer.start()

def pcie_recv_open():
    if 'r32' not in locals() or r32.closed == True:
        try:
            info("Open PCIE receiving channel")
            # r32 = open('\\\\.\\xillybus_read_32' ,'rb')
            r32 = os.open("/dev/xillybus_mua_32", os.O_RDONLY)
            info("PCIE receiving channel is opened")
            return r32
        except:
            info("PCIE receiving channel cannot be opened")
            exit(0)
    else:
        r32.close()
        try:
            info("Open PCIE receiving channel")
            # r32 = open('\\\\.\\xillybus_read_32' ,'rb')
            r32 = os.open("/dev/xillybus_mua_32", os.O_RDONLY)
            info("PCIE receiving channel is opened")
            return r32
        except:
            info("PCIE receiving channel cannot be opened")
            exit(0)

def write_data(shared_arr, write_conn):
    pass
    # f = open('S:/PCIE_.bin','ab+') 
    # data = tonumpyarray(shared_arr)
    # while True:
    #     write_conn.recv()
        # with shared_arr.get_lock():
        # data[:] = tonumpyarray(shared_arr)
        # bytes2write = bytearray(data.astype(np.float32))
        # f.write(bytes2write)
    # f.write(''.join(buff))
    # write_conn.recv()


def get_data(shared_arr, n, read_conn):
    '''
    A daemon process dedicated on reading data from PCIE and update
    the shared memory with other processors: shared_arr 
    '''
    # r32 = pcie_recv_open()
    r32 = io.open('/dev/xillybus_mua_32', 'rb')
    r32_buf = io.BufferedReader(r32)
    # r32 = os.open('/dev/xillybus_mua_32', os.O_RDONLY)
    # buff = []
    # q.put(1)
    # f = open('/tmp_data/pcie.bin','wb') 
    fd = os.open("/tmp_data/pcie.bin", os.O_CREAT | os.O_WRONLY | os.O_NONBLOCK)
    data = tonumpyarray(shared_arr)
    num = 0
    _size = n*32*4  # 32 channels, 4 bytes/sample
    total_size = 0
    
    while True:
        # with shared_arr.get_lock():
        # tic = ptime.time() * 1000
        # info('reading {0} data points'.format(n))
        buf = r32_buf.read(_size)
        # buf = os.read(r32, _size)
        # f.write(buf)
        os.write(fd, buf)
        data[:] = np.frombuffer(buf, dtype='i4')
        # toc = ptime.time() * 1000
        # print toc-tic


def gen_data(shared_arr, n, lock):
    info("Guassian generator starts ------>")
    time.sleep(0.03)
    while True:
        # with shared_arr.get_lock():
        # data = tonumpyarray(shared_arr)
        # data    = shared_arr.numpy()
        lock.acquire()
        data = torch.from_numpy((np.random.randn(n*32) * 50))
        shared_arr[:] = data
        lock.release()
        # time.sleep(0.0001)


def daemon_process_run(read_conn, write_conn, lock, testcase=0):
    '''
    Two test case:
    1. generated data from background process
    2. read data from PCIE background process
    '''
    # testcase 1: Generate data from background process
    if testcase == 0:
        read_proc = Process(target=gen_data, args=(shared_arr, npts, lock))
    # testcase 2: Read data from background process
    elif testcase == 1:
        read_proc = Process(target=get_data, args=(shared_arr, npts, read_conn))
        write_proc= Process(target=write_data, args=(shared_arr,write_conn))
    read_proc.daemon = True
    if testcase == 1:
        write_proc.daemon = True
    read_proc.start()
    if testcase == 1:
        write_proc.start()
    # write_proc.join()

def log_start(level=logging.INFO):
    logger = mp.log_to_stderr()
    logger.setLevel(level)    #DEBUG, INFO, WARNING, ERROR, CRITICAL
    logger.propagate  = False

def main_read():
    log_start(level=logging.INFO)
    read_conn, write_conn = Pipe()
    daemon_process_run(read_conn=read_conn, write_conn=write_conn, testcase=1)   # 0 for guassian, 1 for PCIE
    canvas.show()
    app.run()

if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1:
        # main_read()
        log_start(level=logging.INFO)
        # multiprocessing.freeze_support()
        read_conn, write_conn = Pipe()
        daemon_process_run(read_conn=read_conn, write_conn=write_conn, lock=lock, testcase=0)   # 0 for guassian, 1 for PCIE
        wview.show()
        app.run()

# To see the save file: #####################
# from Binload import Binload
# bf = Binload()
# bf.load('PCIE.bin',file_format='float32')
# bf.plot(n=(0,10000),chNo=4)
#############################################
