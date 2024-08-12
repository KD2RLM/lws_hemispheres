#!/usr/bin/env python
"""
Figure_2a_SuperDARN_map.py
Nathaniel A. Frissell
February 2024

This script is used to generate Figure 2b of the Frissell et al. (2024)
GRL manuscript on multi-instrument measurements of AGWs, MSTIDs, and LSTIDs.
"""

import os
import shutil
import datetime
import numpy as np
import scipy as sp
import pandas as pd

import matplotlib as mpl
from matplotlib import patheffects
from matplotlib import pyplot as plt

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
from cartopy.feature.nightshade import Nightshade

import pickle

#from hamsci_psws import geopack

import pydarn
from pyDARNmusic import load_fitacf
import pyDARNmusic

from pydarn import (Re, time2datetime, Coords, SuperDARNRadars,RangeEstimation)

import mstid
import merra2AirsMaps
import ham_spot_plot
import raytrace_and_plot

Re = 6371 # Radius of the Earth in km

mpl.rcParams['font.size']           = 22
mpl.rcParams['font.weight']         = 'bold'
mpl.rcParams['axes.labelweight']    = 'bold'
mpl.rcParams['axes.labelsize']      = 18
mpl.rcParams['axes.grid']           = True
mpl.rcParams['grid.linestyle']      = ':'
mpl.rcParams['figure.figsize']      = np.array([15, 8])
mpl.rcParams['axes.xmargin']        = 0

def prep_dir(path,clear=False):
    if clear:
        if os.path.exists(path):
            shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)

def time_vector(sTime,eTime,timedelta=datetime.timedelta(minutes=2),**kwargs):
    tvec = [sTime]
    while tvec[-1] < eTime:
        tvec.append(tvec[-1] + timedelta)

    return tvec

def fan_plot(dataObject,
    dataSet                 = 'active',
    time                    = None,
    axis                    = None,
    scale                   = None,
    autoScale               = False,
    plotZeros               = False,
    markCell                = None,
    markBeam                = None,
    markBeam_dict           = {'color':'white','lw':2},
    plotTerminator          = True,
    parallels_ticks         = None,
    meridians_ticks         = None,
    cmap                    = None,
    plot_cbar               = True,
    projection              = ccrs.PlateCarree(),
    plot_fov                = True,
    **kwArgs):

    from pydarn import (SuperDARNRadars, Hemisphere)

    # Make some variables easier to get to...
    currentData = pyDARNmusic.utils.musicUtils.getDataSet(dataObject,dataSet)
    metadata    = currentData.metadata
    latFull     = currentData.fov["latFull"]
    lonFull     = currentData.fov["lonFull"]
    sdate       = currentData.time[0]
    coords      = metadata['coords']
    stid        = metadata['stid']
    radar       = metadata['code'].strip()

    # Get center of FOV.
    # Determine center beam.
    ctrBeamInx  = len(currentData.fov["beams"])/2
    ctrGateInx  = len(currentData.fov["gates"])/2
    ctrLat      = currentData.fov["latCenter"][int(ctrBeamInx),int(ctrGateInx)]
    ctrLon      = currentData.fov["lonCenter"][int(ctrBeamInx),int(ctrGateInx)]

    # Translate parameter information from short to long form.
    paramDict = pyDARNmusic.utils.radUtils.getParamDict(metadata['param'])
    if 'label' in paramDict:
        param     = paramDict['param']
        cbarLabel = paramDict['label'] + ' [{!s}]'.format(paramDict['unit'])
    else:
        param = 'width' # Set param = 'width' at this point just to not screw up the colorbar function.
        cbarLabel = metadata['param'] + ' [{!s}]'.format(paramDict['unit'])

    # Set colorbar scale if not explicitly defined.
    if(scale is None):
        if autoScale:
            sd          = sp.nanstd(np.abs(currentData.data),axis=None)
            mean        = sp.nanmean(np.abs(currentData.data),axis=None)
            scMax       = np.ceil(mean + 1.*sd)
            
            if np.min(currentData.data) < 0:
                scale   = scMax*np.array([-1.,1.])
            else:
                scale   = scMax*np.array([0.,1.])
        else:
            if 'range' in paramDict:
                scale = paramDict['range']
            else:
                scale = [-200,200]

    if stid:
        radar_lat = SuperDARNRadars.radars[stid].hardware_info.geographic.lat
        radar_lon = SuperDARNRadars.radars[stid].hardware_info.geographic.lon

    fig   = axis.get_figure()

    # Figure out which scan we are going to plot...
    if time is None:
        timeInx = 1
    else:
        timeInx = (np.where(currentData.time >= time))[0]
        if np.size(timeInx) == 0:
            timeInx = -1
        else:
            timeInx = int(np.min(timeInx))
            

    # do some stuff in map projection coords to get necessary width and height of map
    lonFull,latFull = (np.array(lonFull)+360.)%360.,np.array(latFull)

    goodLatLon  = np.logical_and( np.logical_not(np.isnan(lonFull)), np.logical_not(np.isnan(latFull)) )
    goodInx     = np.where(goodLatLon)
    goodLatFull = latFull[goodInx]
    goodLonFull = lonFull[goodInx]

    # Plot the SuperDARN data!
    ngates = np.shape(currentData.data)[2]
    nbeams = np.shape(currentData.data)[1]
    data  = currentData.data[timeInx,:,:]
    verts = []
    scan  = []
    # data  = currentData.data[timeInx,:,:]
    goodBmRg=[]
    geo = ccrs.Geodetic()
    for bm in range(nbeams):
        for rg in range(ngates):
            if goodLatLon[bm,rg] == False: continue
            if np.isnan(data[bm,rg]): continue
            if data[bm,rg] == 0 and not plotZeros: continue
            goodBmRg.append((bm,rg))
            scan.append(data[bm,rg])
            x1,y1 = projection.transform_point(lonFull[bm+0,rg+0],latFull[bm+0,rg+0],geo)
            x2,y2 = projection.transform_point(lonFull[bm+1,rg+0],latFull[bm+1,rg+0],geo)
            x3,y3 = projection.transform_point(lonFull[bm+1,rg+1],latFull[bm+1,rg+1],geo)
            x4,y4 = projection.transform_point(lonFull[bm+0,rg+1],latFull[bm+0,rg+1],geo)
            verts.append(((x1,y1),(x2,y2),(x3,y3),(x4,y4),(x1,y1)))
    
    if cmap is None:
        cmap    = mpl.cm.jet
    bounds  = np.linspace(scale[0],scale[1],256)
    norm    = mpl.colors.BoundaryNorm(bounds,cmap.N)

    pcoll = mpl.collections.PolyCollection(np.array(verts),edgecolors='face',closed=False,cmap=cmap,norm=norm,zorder=6001)
    pcoll.set_array(np.array(scan))
    axis.add_collection(pcoll,autolim=False)

    result  = {}
    result['pcoll']     = pcoll
    result['metadata']  = metadata
    result['cbarLabel'] = cbarLabel

    return result

def load_data(load_sTime=None,load_eTime=None,fovModel='GS',
        cache_base='cache',clear_cache=False,
        data_dir='/data/sd-data',use_preprocessed = False,**kwargs):
    radars_dct  = {}
    radars_dct['pgr'] = {}
    radars_dct['sas'] = {}
    radars_dct['kap'] = {}
    radars_dct['gbr'] = {}
    radars_dct['cvw'] = {'fov_beams':(3,25)}
    radars_dct['cve'] = {'fov_beams':(0,21)}
    radars_dct['fhw'] = {}
    radars_dct['fhe'] = {}
    radars_dct['bks'] = {'fov_beams':(3,25)}
    radars_dct['wal'] = {}

    if load_sTime is None:
        load_sTime = kwargs.get('sTime')

    if load_eTime is None:
        load_eTime = kwargs.get('eTime')

    cache_dir = os.path.join(cache_base,data_dir.lstrip('/'))
    prep_dir(cache_dir,clear=clear_cache)

    for radar,dct in radars_dct.items(): 
        if use_preprocessed:
            dataObj         = mstid.more_music.get_dataObj(radar,load_sTime,load_eTime,data_path='mstid_data/mstid_index')
        else:
            cache_fname = '{!s}.{!s}-{!s}.p'.format(radar,load_sTime.strftime('%Y%m%d.%H%M'),load_eTime.strftime('%Y%m%d.%H%M'))
            cache_fpath  = os.path.join(cache_dir,cache_fname)

            if not os.path.exists(cache_fpath):
                gscat                   = 1 # Ground scatter only.
                beam_limits             = (None, None)
                gate_limits             = (0,60)
                interp_resolution       = 60.
                filter_numtaps          = 101.
                bad_range_km            = None

                dataObj = mstid.more_music.create_music_obj(radar.lower(), load_sTime, load_eTime
                    ,beam_limits                = beam_limits
                    ,gate_limits                = gate_limits
                    ,interp_resolution          = interp_resolution
                    ,filterNumtaps              = filter_numtaps 
                    ,srcPath                    = None
                    ,fovModel                   = fovModel
                    ,gscat                      = gscat
                    ,fitacf_dir                 = data_dir
                    )

                if len(dataObj.get_data_sets()) == 0:
                    dataObj = None
                else:
                    gate_limits = mstid.more_music.auto_range(radar,load_sTime,load_eTime,dataObj,bad_range_km=bad_range_km)

                    pyDARNmusic.boxcarFilter(dataObj)
#                    pyDARNmusic.defineLimits(dataObj,gateLimits=gate_limits)

                    dataObj.active.applyLimits()

                    pyDARNmusic.beamInterpolation(dataObj,dataSet='limitsApplied')
                    pyDARNmusic.determineRelativePosition(dataObj)

                    pyDARNmusic.timeInterpolation(dataObj,timeRes=interp_resolution)
                    pyDARNmusic.nan_to_num(dataObj)

                    mstid.more_music.calculate_terminator_for_dataSet(dataObj)

                with open(cache_fpath,'wb') as fl:
                    pickle.dump(dataObj,fl)

            else:
                print('Using cached file: {!s}'.format(cache_fpath))
                with open(cache_fpath,'rb') as fl:
                    dataObj = pickle.load(fl)

        if dataObj is not None:
            print('Loaded: {!s}'.format(radar))
        else:
            print('NO DATA for {!s}'.format(radar))
        dct['dataObj']  = dataObj
    return radars_dct

def get_stid(radar):
    """
    Get the radar stid given the radar abbreviation.
    """

    for stid,rdr_tpl in pydarn.SuperDARNRadars.radars.items():
        if radar == rdr_tpl.hardware_info.abbrev:
            return stid

def plot_radar_fov(radar,ax,time,fovModel='GS',fov_ranges=(0,50),fov_beams=None,rsep=45,frang=180,fov_zorder=6000):
    """
    rsep:  Range Seperation (km) (default: 45 km)
    frang: Distance to first range gate (km) (default: 45 km)
    """
    #Calculate the field of view if it has not yet been calculated.
    stid    = get_stid(radar)
    coords  = pydarn.Coords.GEOGRAPHIC
    radar_lat = SuperDARNRadars.radars[stid].hardware_info.geographic.lat
    radar_lon = SuperDARNRadars.radars[stid].hardware_info.geographic.lon

    if fovModel == "GS":
        range_estimation = pydarn.RangeEstimation.HALF_SLANT
    else:
        range_estimation=RangeEstimation.SLANT_RANGE

    latFull, lonFull = coords(stid=stid,rsep=rsep,frang=frang,
                        gates=fov_ranges, date=time,range_estimation=range_estimation)

    if fov_beams is not None:
        beam_min = np.min(fov_beams)
        beam_max = np.max(fov_beams)

        latFull = latFull[:,beam_min:beam_max]
        lonFull = lonFull[:,beam_min:beam_max]

    # Left Edge
    xx = lonFull[0,:]
    yy = latFull[0,:]
    ax.plot(xx,yy,color='k',transform=ccrs.PlateCarree(),zorder=fov_zorder)

    # Right Edge
    xx = lonFull[-1,:]
    yy = latFull[-1,:]
    ax.plot(xx,yy,color='k',transform=ccrs.PlateCarree(),zorder=fov_zorder)

    # Bottom Edge 
    xx = lonFull[:,0]
    yy = latFull[:,0]
    ax.plot(xx,yy,color='k',transform=ccrs.PlateCarree(),zorder=fov_zorder)

    # Top Edge
    xx = lonFull[:,-1]
    yy = latFull[:,-1]
    ax.plot(xx,yy,color='k',transform=ccrs.PlateCarree(),zorder=fov_zorder)

    # Radar Location
    ax.scatter(radar_lon,radar_lat,marker='o',color='k',s=40,transform=ccrs.PlateCarree(),zorder=fov_zorder)
    if radar == 'cvw' or radar == 'fhw':
        ha      = 'right'
        text    = radar.upper() + ' '
    else:
        ha      = 'left'
        text    = ' ' + radar.upper()

    fontdict = {'color':'black','size':18,'weight':'bold'}
    ax.text(radar_lon,radar_lat,text,ha=ha,
            fontdict=fontdict,transform=ccrs.PlateCarree(),
            path_effects=[mpl.patheffects.withStroke(linewidth=2, foreground="white")],
            zorder=fov_zorder+1)

def plot_rtp(radars_dct,sTime,eTime,dataSet='active',output_dir='output',**kwargs):

    nrows   = len(radars_dct)
    ncols   = 1
    figsize = (20,4*nrows)
    fig     = plt.figure(figsize=figsize)

    ax_inx  = 0
    for radar,dct in radars_dct.items():
        beam = dct.get('beam',7)

        ax_inx += 1
        ax  = fig.add_subplot(nrows,ncols,ax_inx)

        fontdict    = {'fontsize':'x-large','weight':'bold'}
        bbox        = {'boxstyle':'round','facecolor':'white','alpha':1.0}
        text        = '{!s} Beam {!s}'.format(radar.upper(),beam)
        ax.text(0.01,0.95,text,va='top',zorder=1000,
                transform=ax.transAxes,fontdict=fontdict,bbox=bbox)

        if ax_inx == 1:
            fmt   = '%Y %b %d %H%M UTC'
            title = '{!s} - {!s}'.format(sTime.strftime(fmt),eTime.strftime(fmt))
            fontdict={'fontsize':'x-large','weight':'bold'}
            ax.set_title(title,fontdict=fontdict)

        dataObj = dct.get('dataObj')
        if dataObj is None:
            continue
        pyDARNmusic.plotting.rtp.musicRTP(dataObj,xlim=(sTime,eTime),axis=ax,dataSet=dataSet,
                beam=beam,plot_info=False,plot_title=False)

    fname   = 'rtp_{!s}-{!s}.png'.format(sTime.strftime('%Y%m%d.%H%M'),eTime.strftime('%Y%m%d.%H%M'))
    fpath   = os.path.join(output_dir,fname)
    fig.savefig(fpath,bbox_inches='tight')
    plt.close(fig)
    print(fpath)

def plot_map_ax(fig,radars_dct,time,dataSet='active',fovModel='GS',
                    panel_rect          = [0,0,1,1],
                    map_wd              = 0.70,
                    map_hpad            = 0.05,
                    cb_ht               = 0.75,
                    cb_wd               = 0.10,
                    cb_hpad             = 0.07,
                    extent              = None,
                    projection          = ccrs.Orthographic(-100,70),
                    title_size          = None,
                    cbar_ticklabel_size = None,
                    cbar_label_size     = None,
                    SD_scale            = (0,30),
                    AIRS_GWv_scale      = (0.,0.8),
                    **kwargs):
    """
    panel_rect: rectangle in figure coordinatesdefining entire area used by 
                this panel, including colorbars
                [x00, y00, width, height]
    map_wd:     map subpanel width as fraction of panel
    map_hpad:   horizonal padding of map subpanel as fraction of panel
    cb_wd:      colorbar subpanel width as fraction of panel
    cb_hpad:    horizonal padding of colorbar subpanel as fraction of panel
    cb_ht:      colorbar subpanel height as fraction of panel
    """

    # Make panel rectangle values easy to get to.
    p_x00   = panel_rect[0]
    p_y00   = panel_rect[1]
    p_wd    = panel_rect[2]
    p_ht    = panel_rect[3]

    # Scale map and cb values to panel
    _map_wd      = map_wd*p_wd
    _map_hpad    = map_hpad*p_wd

    _cb_ht       = cb_ht*p_ht
    _cb_wd       = cb_wd*p_wd
    _cb_hpad     = cb_hpad*p_wd

    # Calculate specific x and y coordinates
    GNSS_x00    = _map_wd 
    SD_x00      = GNSS_x00 + _cb_wd
    AIRS_x00    = SD_x00   + _cb_wd
    cb_y00      = p_y00+(p_ht-_cb_ht)/2.

    # Define actual axes rectangles.
                        # [x00,      y00,    width,             height]
    map_rect            = [p_x00,    p_y00,  _map_wd-_map_hpad, p_ht]
    GNSS_TEC_cbar_rect  = [GNSS_x00, cb_y00, _cb_wd-_cb_hpad,  _cb_ht]
    SD_cbar_rect        = [SD_x00,   cb_y00, _cb_wd-_cb_hpad,  _cb_ht]
    AIRS_GWv_cbar_rect  = [AIRS_x00, cb_y00, _cb_wd-_cb_hpad,  _cb_ht]

    # PANEL A - OVERVIEW MAP ####################################################### 
    ax  = fig.add_axes(map_rect,projection=projection)
    if extent is not None:
        ax.set_extent(extent,crs=ccrs.PlateCarree())

    # Plot Map Features - Coastlines #######
    ax.coastlines(color='0.7')
    ax.add_feature(cfeature.LAND, color='lightgrey')
    ax.add_feature(cfeature.OCEAN, color = 'white')
    ax.add_feature(Nightshade(time, alpha=0.2))
    ax.gridlines(draw_labels=['left','bottom'])

    # Plot SuperDARN FOVs and Data #########
    for radar,dct in radars_dct.items():
        # SuperDARN FOVs
        fov_ranges  = dct.get('fov_ranges',(0,60))
        fov_beams   = dct.get('fov_beams',None)
        plot_radar_fov(radar,ax,time=time,fovModel=fovModel,
                fov_ranges=fov_ranges,fov_beams=fov_beams)

        dataObj = dct.get('dataObj')
        if dataObj is None:
            # Don't plot data if None exists.
            continue

        # Plot SuperDARN Data
        result = fan_plot(dataObj,dataSet=dataSet,
                axis=ax,projection=projection,time=time,scale=SD_scale)

    # GNSS aTEC
    hsp = rd.get('hsp')
    map_data    = hsp.geo_hist.copy()
    map_data    = np.log10(map_data)
    tf          = np.isfinite(map_data)
    map_data.values[~tf]   = np.nan
    lon_key = map_data.attrs['xkey']
    lat_key = map_data.attrs['ykey']
    xx  = map_data[lon_key].values
    yy  = map_data[lat_key].values
    zz  = map_data.values.T
    tec_mpbl    = ax.contourf(xx,yy,zz,levels=30,cmap=mpl.cm.gray,transform=ccrs.PlateCarree())

    # Plot aTEC Colorbar
    cax  = fig.add_axes(GNSS_TEC_cbar_rect)
    cax.grid(False)
#    tec_cbar_ticks = np.arange(GNSS_TEC_scale[0],GNSS_TEC_scale[1]+0.5,0.5)
#    tec_cbar = fig.colorbar(tec_mpbl,cax=cax,ticks=tec_cbar_ticks)
    tec_cbar = fig.colorbar(tec_mpbl,cax=cax)
    cbar_fd  = {}
    if cbar_label_size is not None:
        cbar_fd.update({'size':cbar_label_size})
    tec_cbar.set_label('GNSS aTEC',fontdict=cbar_fd)
    if cbar_ticklabel_size is not None:
        for ytl in cax.get_yticklabels():
            ytl.set_size(cbar_ticklabel_size)

    # SuperDARN Colorbar
    cax         = fig.add_axes(SD_cbar_rect)
    cax.grid(False)
    cbar_ticks  = np.arange(SD_scale[0],SD_scale[1]+5,5)
    cbar        = fig.colorbar(result['pcoll'],cax=cax,ticks=cbar_ticks,extend='max')
    cbar.set_label('SuperDARN ' + result['cbarLabel'],fontdict=cbar_fd)
    if cbar_ticklabel_size is not None:
        for ytl in cax.get_yticklabels():
            ytl.set_size(cbar_ticklabel_size)

    if 'gscat' in result['metadata']:
        if result['metadata']['gscat'] == 1:
            cbar.ax.text(0.5,-0.075,'Ground\nScatter Only',ha='center',fontsize='x-small',transform=cbar.ax.transAxes)

    txt = 'Coordinates: ' + result['metadata']['coords'] +', Model: ' + result['metadata']['model']
    ax.text(1.01, 0, txt,
              horizontalalignment='left',
              verticalalignment='bottom',
              rotation='vertical',
              size='x-small',
              weight='bold',
              transform=ax.transAxes)

    # AIRS GW Variance #####################
    mca = kwargs.get('mca')
    if mca is not None:
        date = datetime.datetime(time.year,time.month,time.day)
        if date in mca.get_dates():
            # Plot AIRS Data
            mca_result      = mca.plot_ax(ax=ax,date=date,vmin=AIRS_GWv_scale[0],
                                vmax=AIRS_GWv_scale[1],cmap='RdPu',gridlines=False,coastlines=False)

            # Plot AIRS Colorbar
            cax  = fig.add_axes(AIRS_GWv_cbar_rect)
            cax.grid(False)
            AIRS_cbar_ticks = np.arange(AIRS_GWv_scale[0],AIRS_GWv_scale[1]+0.1,0.1)
            mca_cbar = fig.colorbar(mca_result['cbar_pcoll'],cax=cax,ticks=AIRS_cbar_ticks)
            mca_cbar.set_label(mca_result['cbar_label'],fontdict=cbar_fd)
            if cbar_ticklabel_size is not None:
                for ytl in cax.get_yticklabels():
                    ytl.set_size(cbar_ticklabel_size)

    # Star for approx center of GW Hotspot # 
    ax.scatter([112],[60],s=500,marker='*',ec='black',fc='yellow',
            zorder=10000,transform=ccrs.PlateCarree())

    # Axis Title ########################### 
    fontdict    = {'size':'x-large','weight':'bold'}
    title       = time.strftime('%d %b %Y %H%M UTC')
    if title_size is not None:
        fontdict.update({'size':title_size})
    ax.set_title(title,fontdict=fontdict)

def plot_map(radars_dct,time,figsize=(18,14),output_dir='output',**kwargs):

    fig     = plt.figure(figsize=figsize)
    plot_map_ax(fig,radars_dct,time,**kwargs)

    # Save Figure ##################################################################
    fname = 'map_{!s}.png'.format(time.strftime('%Y%m%d.%H%M'))
    fpath   = os.path.join(output_dir,fname)
    fig.savefig(fpath,bbox_inches='tight')
    plt.close(fig)
    print(fpath)

def plot_fig_rects(fig,rects,vpad=0,color='k',lw=2,fill=False,
        plot_rects=True,plot_outer=True,plot_names=True,**kwargs):
    """
    Print boundaries directly on figure for a dictionary
    of rectangles.

    This is useful for checking where the boundaries of
    axis areas actually are.
    """
    for name,rect in rects.items():
        # Make sure you do not change the original rectangle.
        rect    = rect.copy()

        # Add equal vertical padding to top and bottom.
        rect[1] -= vpad/2.
        rect[3] += vpad

        xy      = (rect[0], rect[1])
        width   = rect[2]
        height  = rect[3]

        if plot_rects:
            fig.patches.extend([plt.Rectangle(xy,width,height,
                color=color,lw=lw,fill=fill,
                transform=fig.transFigure, figure=fig,**kwargs)])

        if plot_names:
            tx  = rect[0] + 0.01
            ty  = rect[1] + rect[3] - 0.01

#            if name == 'c':
#                tx += 0.010
            fontdict    = {'weight':'bold','size':'x-large'}
            fig.text(tx,ty,'({!s})'.format(name),fontdict=fontdict,va='top')

    if plot_outer:
        rect = [0,0,1,1]
        xy      = (rect[0], rect[1])
        width   = rect[2]
        height  = rect[3]

        fig.patches.extend([plt.Rectangle(xy,width,height,
            color=color,lw=lw,fill=fill,ls=':',
            transform=fig.transFigure, figure=fig,**kwargs)])

def figure2(radars_dct,time,hsp,RTaP,figsize=(23,27),output_dir='output',**kwargs):
    fig     = plt.figure(figsize=figsize)
    # Font Control ################################################################# 
    map_title_size          = 'x-large'
    map_cbar_ticklabel_size = 'small'
    map_cbar_label_size     = 'medium'

    ham_title_size          = 'x-large'
    ham_ticklabel_size      = 'small'
    ham_label_size          = 'medium'
    ham_cbar_ticklabel_size = 'small'
    ham_cbar_label_size     = 'medium'

    rt_title_size           = 'medium'
    rt_ticklabel_size       = 'small'
    rt_label_size           = 'medium'
    rt_cbar_ticklabel_size  = 'small'
    rt_cbar_label_size      = 'medium'

    # Panel Positioning ############################################################
    map_ht      = 0.55
    ham_ht      = 0.20
    rt_ht       = (1- map_ht - ham_ht)
    vpad        = 0.055

    ham_hpad    = 0.100
    ham_map_wd  = 0.380
    ham_ts_wd   = 1 - ham_map_wd - ham_hpad

    ham_map_x00 = 0.
    ham_ts_x00  = ham_map_x00 + ham_map_wd

    r0_y00  = 1 - map_ht
    r1_y00  = r0_y00 - ham_ht
    r2_y00  = r1_y00 - rt_ht


    rects   = {}
                # [x00,  y00, width, height]
    rects['a']   = [       0.00, r0_y00,       1.00, map_ht]
    rects['b']   = [ham_map_x00, r1_y00, ham_map_wd, ham_ht]
    rects['c']   = [ ham_ts_x00, r1_y00,  ham_ts_wd, ham_ht]
    rects['d']   = [       0.00, r2_y00,       1.00,  rt_ht]
    for key,rect in rects.items():
        rect[3] -= vpad

    plot_fig_rects(fig,rects,vpad=vpad,plot_rects=False,plot_outer=False)

    rects['c'][0] += ham_hpad

    # Plot Panel (a) Map ###########################################################
    rect    = rects['a']
    dims    = {}
    dims['map_wd']              = 0.75
    dims['map_hpad']            = 0.00
    dims['cb_ht']               = 0.75
    dims['cb_wd']               = (1-dims['map_wd'])/3.
    dims['cb_hpad']             = 0.70*dims['cb_wd']
    dims['extent']              = (0,360,20,90)
    dims['projection']          = ccrs.Orthographic(-100,70.0)
    dims['title_size']          = map_title_size
    dims['cbar_ticklabel_size'] = map_cbar_ticklabel_size
    dims['cbar_label_size']     = map_cbar_label_size
    plot_map_ax(fig,radars_dct,time,panel_rect=rect,**dims,**kwargs)

    # Plot Panel (b) Ham Radio Map #################################################
    ham_fonts   = {}
    ham_fonts['title_size']          = ham_title_size
    ham_fonts['ticklabel_size']      = ham_ticklabel_size
    ham_fonts['label_size']          = ham_label_size
    ham_fonts['cbar_ticklabel_size'] = ham_cbar_ticklabel_size
    ham_fonts['cbar_label_size']     = ham_cbar_label_size

    rect    = rects['b']
    hsp.plot_map_ax(fig,panel_rect=rect,**ham_fonts)

    # Plot Panel (c) Ham Radio Time Series #########################################
    rect            = rects['c']
    ax              = fig.add_axes(rect)
    hspd            = {}
    hspd['xlim']    = (date + datetime.timedelta(hours=13),
                       date + datetime.timedelta(hours=23))
    hspd['ylim']    = (750,2250)
    hspd['cb_pad']  = 0.01
    hspd.update(ham_fonts)
    result          = hsp.plot_timeSeries_ax(ax,**hspd)

    # Plot Panel (d) Ray Trace Diagram #############################################
    rt_fonts   = {}
    rt_fonts['title_size']          = rt_title_size
    rt_fonts['ticklabel_size']      = rt_ticklabel_size
    rt_fonts['label_size']          = rt_label_size
    rt_fonts['cbar_ticklabel_size'] = rt_cbar_ticklabel_size
    rt_fonts['cbar_label_size']     = rt_cbar_label_size

    rect            = rects['d']
    RTaP.plot_ax(fig=fig,panel_rect=rect,**rt_fonts)

    # Save Figure ##################################################################
    fname = 'map_{!s}.png'.format(time.strftime('%Y%m%d.%H%M'))
    fpath   = os.path.join(output_dir,fname)
    fig.savefig(fpath,bbox_inches='tight')
    plt.close(fig)
    print(fpath)

if __name__ == '__main__':
    rd = {}

    rd['output_dir'] = 'output/Fig2_SuperDARN_Map'
    prep_dir(rd['output_dir'],clear=True)

    date                = datetime.datetime(2018,12,15)
    rd['sTime']         = date + datetime.timedelta(hours=17) 
    rd['eTime']         = date + datetime.timedelta(hours=21) 
    rd['time']          = date + datetime.timedelta(hours=20) 

    rd['mca']           = merra2AirsMaps.Merra2AirsMaps()
    rd['hsp']           = ham_spot_plot.HamSpotPlot(date)

    iono_nc = 'data/iri_tid_1000km/20181512.2000-20181512.2000_TX__profile.nc'
    rd['RTaP']          = raytrace_and_plot.RayTraceAndPlot(iono_nc)
  
    rd['fovModel']      = 'GS'
    rd['data_dir']      = '/data/sd-data'
#    rd['data_dir']      = '/data/sd-data_fitexfilter'
    rd['clear_cache']   = False
    rd['radars_dct']    = load_data(**rd)

    rd['dataSet']       = 'originalFit'
    
#    plot_rtp(**rd)
#    plot_map(**rd)

    figure2(**rd)
