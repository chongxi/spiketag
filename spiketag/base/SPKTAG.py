from .MUA import MUA
from .SPK import SPK
from .FET import FET
from .CLU import CLU
import numpy as np
import json
import pickle


class SPKTAG(object):
	def __init__(self, probe=None, pivotal=None, spk=None, fet=None, clu=None, filename=None):
		'''
		pivotal: mua.pivotal_pos (a numpy array), s1st row is time, 2nd row is ch
		spk    : spk object
		fet    : fet object
		clu    : dictionary of clu object (each item is a channel based clu object)
		'''
		if filename is not None: # load from file
			self.fromfile(filename)
		elif probe is not None and pivotal is not None:                    # construct
		        self.probe  = probe	
			arg_piv     = np.lexsort((pivotal[0], pivotal[1])) 
			piv         = pivotal[:, arg_piv]
			self.nspk   = arg_piv.shape[0]
			self.t      = piv[0]
			self.ch     = piv[1]
			self.spk    = spk 
			self.fet    = fet 
			self.clu    = clu 
			self.meta   = {}
			self.spklen = spk.spklen
			self.fetlen = fet.fetlen
			self.dtype  = [('t', 'int32'), 
			               ('ch','int32'),  
			               ('spk', 'f4', (self.spklen, self.probe.len_group)), 
			               ('fet','f4',(self.fetlen,)),
			               ('clu','int32')]
                        self.build_meta()
			self.build_spktag()
		else:
			pass


	def build_meta(self):
                self.meta["probe"] = pickle.dumps(self.probe)
		self.meta["nspk"] = self.nspk
		self.meta["fetlen"] = self.fetlen
		self.meta["spklen"] = self.spklen


	def build_spktag(self):
		spktag = np.zeros(self.nspk, dtype=self.dtype)
		spktag['t']  = self.t
		spktag['ch'] = self.ch
		for chNo in range(self.probe.n_ch):
		    spktag['spk'][spktag['ch']==chNo] = self.spk[chNo]
		    spktag['fet'][spktag['ch']==chNo] = self.fet[chNo]        
		    spktag['clu'][spktag['ch']==chNo] = self.clu[chNo].membership
		self.spktag = spktag


	def update(self, spk, fet, clu):
		self.spk = spk
		self.fet = fet
		self.clu = clu
		self.build_spktag()	


	def tofile(self, filename):
		with open(filename+'.meta', 'w') as metafile:
			json.dump(self.meta, metafile)
		self.spktag.tofile(filename)


	def fromfile(self, filename):
		with open(filename+'.meta', 'r') as metafile:
			self.meta = json.load(metafile)
                self.probe = pickle.loads(self.meta['probe'])
		self.nspk   = self.meta['nspk']
		self.spklen = self.meta['spklen']
		self.fetlen = self.meta['fetlen']
		self.dtype = [('t', 'int32'), 
		              ('ch','int32'),  
		              ('spk', 'f4', (self.spklen, self.probe.len_group)), 
		              ('fet','f4',(self.fetlen,)),
		              ('clu','int32')]
		self.spktag = np.fromfile(filename, dtype=self.dtype)
		self.t  = self.spktag['t']
		self.ch = self.spktag['ch']


	def tospk(self):
		spkdict = {}
		for ch in range(self.probe.n_ch):
			spkdict[ch] = self.spktag['spk'][self.ch==ch]
		return SPK(spkdict)		


	def tofet(self):
		fetdict = {}
		for ch in range(self.probe.n_ch):
			fetdict[ch] = self.spktag['fet'][self.ch==ch]
		return FET(fetdict)		


	def toclu(self):
		cludict = {}
		for ch in range(self.probe.n_ch):
			cludict[ch] = CLU(self.spktag['clu'][self.ch==ch])
		return cludict
