import numpy as np
import struct
from binascii import hexlify


########### transformation and template #############
def write_tat_32(offset, v, dtype='<i4', binpoint=14):
    addr = offset * 4
    w32 = open('/dev/xillybus_template_32', 'wb')
    value = to_fixed_point(v, dtype, binpoint)
    # print 'mem content:', hexlify(value)
    w32.seek(addr)
    w32.write(value)
    w32.close()

def read_tat_32(offset, dtype='<i4', binpoint=14):
    addr = offset * 4
    r32 = open('/dev/xillybus_template_32', 'rb')
    r32.seek(addr)
    hexstring = r32.read(4)
    # print 'mem content:', hexlify(hexstring)
    value = to_value(hexstring, dtype, binpoint)
    r32.close()     
    return value 


########### thr and ch hash #########################
def write_thr_32(offset, v, dtype='<i4', binpoint=14):
    addr = offset * 4
    w32 = open('/dev/xillybus_thr_32', 'wb')
    value = to_fixed_point(v, dtype, binpoint)
    # print 'mem content:', hexlify(value)
    w32.seek(addr)
    w32.write(value)
    w32.close()        

def read_thr_32(offset, dtype='<i4', binpoint=14):
    addr = offset * 4
    r32 = open('/dev/xillybus_thr_32', 'rb')
    r32.seek(addr)
    hexstring = r32.read(4)
    # print 'mem content:', hexlify(hexstring)
    value = to_value(hexstring, dtype, binpoint)
    r32.close()     
    return value 

def thr_reset(nCh):
    for addr in np.arange(nCh):
        write_thr_32(addr, 0, '<i4', 0)
####################################################


############ xillybus_mem_16 #######################
def write_mem_16(offset, v, dtype='<h', binpoint=0):
    addr = offset * 2
    w16 = open('/dev/xillybus_mem_16', 'wb')
    value = to_fixed_point(v, dtype, binpoint)
    # print 'mem content:', hexlify(value)
    w16.seek(addr)
    w16.write(value)
    w16.close()

def read_mem_16(offset, dtype='<h', binpoint=0):
    addr = offset * 2
    r16 = open('/dev/xillybus_mem_16', 'rb')
    r16.seek(addr)
    hexstring = r16.read(2)
    # print 'mem content:', hexlify(hexstring)
    value = to_value(hexstring, dtype, binpoint)
    r16.close() 
    # value is always XXX.0 when binpoint = 0
    return int(value)

def mem_reset(nCh):
    for addr in np.arange(nCh):
        write_reg_16(addr, 0x0000, '<i2', 0)
####################################################


########### quantization ###########################
def to_fixed_point(v, dtype='<i4', binpoint=14):
    v = int(v * 2**binpoint)
    value = hexlify(struct.pack(dtype,v))
    # print value
    return struct.pack(dtype,v)

def to_value(hexstring, dtype='<i4', binpoint=14):
    value = struct.unpack(dtype, hexstring)
    value = float(value[0]) / 2**binpoint
    return value

def to2scomp(v, dtype='i4'):
    value = hexlify(struct.pack(dtype,v))
    return value
#####################################################


def mem_test(addr, value, dt, bp):
    # write mem (32bits)
    print('write {0} to addres {1}'.format(value, addr))
    write_thr_32(addr, value, dtype=dt, binpoint=bp)
    # read mem (32bits)

    print('read v from addres {0}'.format(addr))
    v = read_thr_32(addr, dtype=dt, binpoint=bp)
    print('v =',v)

    # verification
    if abs(value - v) < 2**-bp:
        print('memory test pass\n')
    else:
        print('memory test fail\n')

def test1():
    addr  = 516
    value = -170
    dt = '<i4'
    bp = 0
    mem_test(addr, value, dt, bp)    

def test2():
    value = 0
    dt = '<i4'
    bp = 0
    for addr in np.arange(32):
        value += 1
        mem_test(addr, value, dt, 0)


if __name__ == '__main__':
    # test 1: write single address
    test1()
    # mem_reset(1024)
    # test2()
    # test 2: write multiple address
    # test2()

#  `!hexdump -C -v -n 512 \\.\xillybus_thr_32` to view memory block
# a6e22e
