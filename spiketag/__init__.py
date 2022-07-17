# from .spiketag import check_fpga

__version__ = '0.1.0'
import mkl
mkl.set_num_threads(1) #prevent the conflicts on the multicore computing (numba, pytorch and ipyparallel)

import warnings
warnings.filterwarnings("ignore")


from IPython.core.magic import (Magics, magics_class, line_magic, cell_magic)

def load_ipython_extension(ipython, *args):
    # print('cool')
    # ipython.register_magic_function(FPGA, 'line')
    import asyncio
    code ='''from spiketag.fpga import FPGA\nfrom spiketag.base import probe\nfpga = FPGA()'''
    asyncio.run(ipython.run_code(code))

def FPGA(line):
    print(line)
    from spiketag.fpga import FPGA
    fpga = FPGA()
    return fpga


# @magics_class
# class SPIKETAG_MAGIC(Magics):

#     @line_magic
#     def fpga(self, line):
#         from spiketag.fpga import FPGA
#         fpga = FPGA()
#         return fpga

#     @cell_magic
#     def cadabra(self, line, cell):
#         return line, cell