import sys
import click
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication


@click.group()
def main():
    pass


@main.command()
@click.argument('prbfile', nargs=1)
def prb_check(prbfile, font_size=33):
    '''
    check prb by visualization
    '''
    from spiketag.base import probe
    prb = probe()
    prb.load(prbfile)
    prb.show(font_size)


@main.command()
@click.argument('binfile', nargs=1)
@click.option('--dtype', prompt='dtype', default='int16')
@click.option('--nch', prompt='nch', default='160')
@click.option('--fs', prompt='fs', default='25000')
@click.option('--view', prompt='view', default='y')
def bin_check(binfile, dtype, nch, fs, view):
    '''
    check prb by visualization
    '''
    from spiketag.base import bload
    fs, nch = float(fs), int(nch)
    bf = bload(nCh=nch, fs=fs)
    bf.load(binfile, dtype=dtype)
    if view == 'y':
        from vispy import app
        bf.show()
        app.run()


@main.command()
@click.argument('binaryfile', nargs=2)
@click.option('--nbits', prompt='nbits', default='16')
@click.option('--fs', prompt='fs', default='25000')
@click.option('--src_nch', prompt='src_nch', default='175')
@click.option('--dst_nch', prompt='dst_nch', default='160')
def convert(binaryfile, nbits, fs, src_nch, dst_nch):
    '''
    convert 175 chs open-ephys raw to 160 chs pure raw (16 bits)
    This process is prb ignorant
    '''
    from spiketag.base import bload
    nbits, fs, src_nch, dst_nch = int(nbits), float(fs), int(src_nch), int(dst_nch)
    src_file, sink_file = binaryfile

    bf = bload(nCh=src_nch, fs=fs)
    bf.load(src_file, dtype=np.int16)
    data = bf.npmm.reshape(-1, src_nch)[:, :dst_nch]
    click.echo('converting {} to {}'.format(src_file, sink_file))
    data.tofile(sink_file)
    df = bload(nCh=dst_nch, fs=fs)
    click.echo('convert finished')
    df.load(sink_file, dtype=np.int16)


@main.command()
@click.argument('binaryfile', nargs=2)
@click.option('--nbits', prompt='nbits', default='32')
@click.option('--nch', prompt='nch', default='160')
@click.option('--fs', prompt='fs', default='25000')
def deconvolve(binaryfile, nbits, nch, fs):
    '''
    MUA(32bits) to RAW(16bits) (inverse filter)
    '''
    from spiketag.base import mua_kernel as kernel
    from spiketag.base import bload
    nbits, fs, nch = int(nbits), float(fs), int(nch)
    src_file, sink_file = binaryfile
    if nbits==32:
        datatype = np.int32
    elif nbits==16:
        datatype = np.int16

    bf = bload(nCh=nch, fs=fs)
    bf.load(src_file, dtype=datatype)
    click.echo('deconvolve {} to RAW'.format(src_file))
    bf.deconvolve(kernel)
    data = bf.asarray(binpoint=13)
    data = data.astype(np.int16)
    click.echo('save to {}'.format(sink_file))
    data.tofile(sink_file)
    df = bload(nCh=nch, fs=fs)
    click.echo('deconvolution finished')
    df.load(sink_file, dtype=np.int16)


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
    import mplcursors
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
        ch = ch[t>0.004]
        t =   t[t>0.004]
        c = Counter(ch)
        nspks = np.array([c[ch] for ch in prb.chs]) 
        nspks_max = nspks.max()
        sns.set_context('paper')
        sns.set(font_scale=1.5)
        sns.set_style('white')
        plt.style.use('dark_background')

        fig, ax = plt.subplots(1,2, figsize=(25,15), gridspec_kw = {'width_ratios':[3, 1]})
        sns.scatterplot(t, prb.ch_idx[ch], hue=prb.ch_idx[ch], palette=plt.cm.hsv, s=18, marker='|',
                        legend=False, ax=ax[0])
        ax[0].set_facecolor((.1, .1, .1, .0))
        ax[0].set_ylim(ax[0].get_ylim()[::-1])
        ax[0].set_xlabel('Time(secs)')
        ax[0].set_ylabel('Virtual Channel Number')
        ax[0].set_yticks(np.arange(0,160,8))
        ax[0].invert_yaxis()
        ax[0].set_title('#spikes found on (channels) vs (time)')
        nspks_img = ax[1].imshow(nspks.reshape(-1, prb.group_len), cmap=plt.cm.magma) #ocean_r
        ax[1].grid(which='minor', color='w', linestyle='-', linewidth=2)
        ax[1].set_ylabel('Group Number')
        ax[1].set_xticks(range(prb.group_len))
        ax[1].set_yticks(range(prb.n_group))
        ax[1].set_xlabel('Channels in the Group')
        ax[1].set_title('#spikes found on (channels) in (group)')
        ax[1].invert_yaxis()
        sns.despine()

        for i in range(prb.grp_matrix.shape[0]):
            for j in range(prb.grp_matrix.shape[1]):
                if nspks[prb.grp_matrix[i,j]] < nspks_max*0.7:
                    ax[1].text(j, i, str(prb.grp_matrix[i,j]), color='white', ha='center', va='center', fontsize=10)
                else:
                    ax[1].text(j, i, str(prb.grp_matrix[i,j]), color='black', ha='center', va='center', fontsize=10)

        fig.colorbar(nspks_img)
        plt.tight_layout()
        mplcursors.cursor()
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
    # ctrl.model.sort(clu_method='dpgmm', group_id=0, n_comp=8, max_iter=400)
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


@main.command()
def raster():
    '''
    >>> spiketag raster
    '''
    from spiketag.view import raster_view
    rasview = raster_view()
    try:
        rasview.fromfile('./fet.bin')
        rasview.show()
        rasview.title='spike raster'
        rasview.app.run()
    except:
        print('fail to load fet.bin, check whether has this file under current folder')


@main.command()
def feature():
    '''
    spiketag feature
    '''
    from spiketag.view import grid_scatter3d
    app = QApplication(sys.argv) 
    grid_fetview = grid_scatter3d(5,8)
    grid_fetview.show()
    grid_fetview.from_file('./fet.bin')
    sys.exit(app.exec_())


@main.command()
@click.option('--gui', prompt='raster/fet', default='raster')
def bmi(gui):
    # import os
    # os.remove("./fet.bin")

    if gui == 'raster':
        '''
        >>> spiketag bmi raster
        '''
        from spiketag.res.GUI.BMI_RASTER_GUI import BMI_RASTER_GUI
        app = QApplication(sys.argv) 
        gui = BMI_RASTER_GUI(fet_file='./fet.bin', t_window=10e-3, view_window=10)
        gui.show()
        sys.exit(app.exec_())

    if gui == 'fet':
        '''
        >>> spiketag bmi fet
        '''
        from spiketag.res.GUI.BMI_FET_GUI import BMI_GUI as BMI_FET_GUI
        app = QApplication(sys.argv) 
        gui = BMI_FET_GUI(fet_file='./fet.bin')
        gui.show()
        sys.exit(app.exec_())