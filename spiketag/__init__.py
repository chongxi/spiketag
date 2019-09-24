# from .spiketag import check_fpga

__version__ = '0.1.0'
# from .mvc.Control import Sorter
import mkl
mkl.set_num_threads(1) #prevent the conflicts on the multicore computing (numba, pytorch and ipyparallel)
