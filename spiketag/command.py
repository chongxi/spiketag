import click
import numpy as np
import matplotlib.pyplot as plt


@click.group()
def main():
    pass


@main.command()
@click.argument('binaryfile', nargs=-1)
@click.argument('probefile',  nargs=1)
@click.option('--nbits', prompt='nbits', default='32')
@click.option('--chs', prompt='chs', default='0,128')
@click.option('--time', prompt='time', default='0')
@click.option('--span', prompt='span', default='10')
def view(binaryfile, probefile, nbits, chs, time, span):
    '''
    view raw or mua file with or without spk_info:
    `spiketag view mua.bin spk.bin prb.json`
    '''
    from spiketag.base import probe
    from spiketag.base import MUA 
    from vispy import app
    nbits, time, span = int(nbits), float(time), float(span)
    chs_tbview = slice(*[int(_) for _ in chs.split(',')])
    prb = probe()
    prb.load(probefile)
    if len(binaryfile) == 2:
        mua_filename, spk_filename = binaryfile
        click.echo('loadding {} and {}'.format(mua_filename, spk_filename))
        mua = MUA(mua_filename=mua_filename, spk_filename=spk_filename,
                  probe=prb, numbytes=nbits//8, scale=False)
        if span == -1: span=None
        mua.show(chs=prb.chs[chs_tbview], span=span, time=time)
        app.run()
    elif len(binaryfile) == 1:
        mua_filename = binaryfile[0]
        click.echo('loadding {}'.format(mua_filename))
        mua = MUA(mua_filename=mua_filename, spk_filename=None,
                  probe=prb, numbytes=nbits//8, scale=False)
        if span == -1: span=None
        mua.show(chs=prb.chs[chs_tbview], span=span, time=time)
        app.run()


@main.command()
@click.argument('binaryfile', nargs=-1)
@click.argument('probefile')
def report(binaryfile, probefile, nbits=32):
    '''
    view raw or mua file with or without spk_info:
    `spiketag view mua.bin spk.bin prb.json`
    '''
    from spiketag.base import probe
    from spiketag.base import MUA 
    from collections import Counter
    import seaborn as sns
    nbits = int(nbits)
    prb = probe()
    prb.load(probefile)
    if len(binaryfile) == 2:
        mua_filename, spk_filename = binaryfile
        click.echo('loadding {} and {}'.format(mua_filename, spk_filename))
        mua = MUA(mua_filename=mua_filename, spk_filename=spk_filename,
                  probe=prb, numbytes=nbits//8, scale=False)

        ## report ch<-->nspks statistics
        spk_info = np.fromfile(spk_filename, np.int32).reshape(-1,2)
        t, ch = spk_info[:,0]/prb.fs, spk_info[:,1]
        c = Counter(ch)
        nspks = np.array([c[ch] for ch in prb.chs]) 

        fig, ax = plt.subplots(1,2, figsize=(25,15), gridspec_kw = {'width_ratios':[3, 1]})
        ax[0].plot(t, prb.ch_idx[ch], '.', markersize=1)
        ax[0].set_ylim(ax[0].get_ylim()[::-1])
        ax[0].set_xlabel('Time(secs)')
        ax[0].set_ylabel('Virtual Channel Number')
        ax[0].set_title('#spikes found on (channels) vs (time)')
        nspks_img = ax[1].imshow(nspks.reshape(-1, prb.group_len), cmap='ocean_r')
        ax[1].grid(which='minor', color='k', linestyle='-', linewidth=2)
        ax[1].set_ylabel('Group Number')
        ax[1].set_xticks(range(prb.group_len))
        ax[1].set_yticks(range(prb.n_group))
        ax[1].set_xlabel('Channels in the Group')
        ax[1].set_title('#spikes found on (channels) in (group)')

        for i in range(prb.grp_matrix.shape[0]):
            for j in range(prb.grp_matrix.shape[1]):
                ax[1].text(j, i, str(prb.grp_matrix[i,j]), color='black', ha='center', va='center', fontsize=8)

        fig.colorbar(nspks_img)
        plt.show()

    elif len(binaryfile) == 1:
        mua_filename = binaryfile[0]
        click.echo('loadding {}'.format(mua_filename))
        mua = MUA(mua_filename=mua_filename, spk_filename=None,
                  probe=prb, numbytes=nbits//8, scale=False)
 

# @click.option('--time_cutoff', prompt='time_cutoff', default='0')
@main.command()
@click.argument('binaryfile', nargs=-1)
@click.argument('probefile')
def sort(binaryfile, probefile):
    '''
    sort without notebook:
    `spiketag sort mua.bin spk.bin prb.json`
    '''
    mua_filename, spk_filename = binaryfile
    click.echo('spiketag-sort: loadding {} and {}'.format(mua_filename, spk_filename))
    import sys
    from PyQt5.QtWidgets import QApplication
    from spiketag.mvc.Control import controller
    from spiketag.base import probe
    prb = probe()
    prb.load(probefile)
    app  = QApplication(sys.argv)
    ctrl = controller(
                      probe = prb,
                      mua_filename=mua_filename, 
                      spk_filename=spk_filename, 
                      binary_radix=13, 
                      scale=False
                      # time_segs=[[0,320]]
                     )
    ctrl.model.sort(clu_method='dpgmm', group_id=0, n_comp=8, max_iter=400)
    ctrl.show()
    sys.exit(app.exec_())


@main.command()
@click.option('--var', prompt='ch_grpNo, ch_hash, ch_ref, thres, scale, shift, pca, vq')
def fpga_check(var):
    '''
    check fpga params:
    `spiketag fpga-check`
    '''
    click.echo('check FPGA')
    from spiketag.fpga import xike_config
    fpga = xike_config()
    if var in dir(fpga):
        exec('print(fpga.{})'.format(var))  


@main.command()
@click.argument('cmd')
def fpga_detector(cmd):
    '''
    set detector params:
    `spiketag fpga-detector thres==-500`
    `spiketag fpga-detector ch_ref==-14`
    '''
    click.echo('set FPGA detector {} '.format(cmd))
    from spiketag.fpga import xike_config
    fpga = xike_config()
    exec('fpga.{}'.format(cmd))  


@main.command()
@click.argument('probefile')
def fpga(probefile):
    '''
    init fpga with probe file:
    `spiketag fpga prb.json`
    '''
    click.echo('init FPGA with probe file {}'.format(probefile))
    from spiketag.fpga import xike_config
    from spiketag.base import probe
    prb = probe()
    prb.load(probefile)    
    prb.show()
    fpga = xike_config(prb)
    click.echo('init FPGA success, check fpga with "spiketag fpga-check"')


@main.command()
@click.argument('notebookfile')
def cp(notebookfile):
    '''
    start sorting using template notebook:
    `spiketag cp sorter`
    `spiketag cp sorter-fpga`
    '''
    click.echo('copy notebook {} to current folder'.format(notebookfile))
    import os
    url = 'https://raw.githubusercontent.com/chongxi/spiketag/master/notebooks/template'
    os.system('wget {}/{}.ipynb'.format(url, notebookfile))
