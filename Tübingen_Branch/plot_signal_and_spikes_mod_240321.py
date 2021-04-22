# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 09:15:36 2021

@author: jonas ort
"""






def plot_analog_stream_channel(analog_stream, channel_idx, from_in_s=0, to_in_s=None, show=True):
    """
    Plots data from a single AnalogStream channel

    :param analog_stream: A AnalogStream object
    :param channel_idx: A scalar channel index (0 <= channel_idx < # channels in the AnalogStream)
    :param from_in_s: The start timestamp of the plot (0 <= from_in_s < to_in_s). Default: 0
    :param to_in_s: The end timestamp of the plot (from_in_s < to_in_s <= duration). Default: None (= recording duration)
    :param show: If True (default), the plot is directly created. For further plotting, use show=False
    """
    # extract basic information
    ids = [c.channel_id for c in analog_stream.channel_infos.values()]
    channel_id = ids[channel_idx]
    channel_info = analog_stream.channel_infos[channel_id]
    sampling_frequency = channel_info.sampling_frequency.magnitude

    # get start and end index
    from_idx = max(0, int(from_in_s * sampling_frequency))
    if to_in_s is None:
        to_idx = analog_stream.channel_data.shape[1]
    else:
        to_idx = min(analog_stream.channel_data.shape[1], int(to_in_s * sampling_frequency))

    # get the timestamps for each sample
    time = analog_stream.get_channel_sample_timestamps(channel_id, from_idx, to_idx)

    # scale time to seconds:
    scale_factor_for_second = Q_(1,time[1]).to(ureg.s).magnitude
    time_in_sec = time[0] * scale_factor_for_second

    # get the signal
    signal = analog_stream.get_channel_in_range(channel_id, from_idx, to_idx)

    # scale signal to µV:
    scale_factor_for_uV = Q_(1,signal[1]).to(ureg.uV).magnitude
    signal_in_uV = signal[0] * scale_factor_for_uV

    # construct the plot
    _ = plt.figure(figsize=(20,6))
    _ = plt.plot(time_in_sec, signal_in_uV)
    _ = plt.xlabel('Time (%s)' % ureg.s)
    _ = plt.ylabel('Voltage (%s)' % ureg.uV)
    _ = plt.title('Channel %s' % channel_info.info['Label'])
    if show:
        plt.show()



def plot_signal_and_spikes_from_bandpassfilteredsignal(bandpassfilteredsignal, 
                                                       spikes, threshold, channellabel, fs, time_in_sec, from_in_s=0, to_in_s=None, tick=40, 
                                                       scale_factor_for_second=1e-06, show=True):
    
    import matplotlib.pyplot as plt
    plt.style.use('seaborn')
    
    to_sec=tick*scale_factor_for_second
    '''
    start_range=max(0, int((from_in_s/tick)/scale_factor_for_second))
    if to_in_s==None:
        stop_range=len(bandpassfilteredsignal)
    else:
        stop_range=min(len(bandpassfilteredsignal), to_in_s/to_sec)
    '''
    
    
    corrector=0
    if time_in_sec[0]!=0:
        corrector+=int(time_in_sec[0]/to_sec)
        spikes_correct=spikes+corrector
    else:
        spikes_correct=spikes
    
    start_range=(max(time_in_sec[0], from_in_s))
    start_range_signal=(max(0, int(from_in_s/to_sec)))
    
    if to_in_s==None:
        stop_range=time_in_sec[-1]
        stop_range_signal=len(bandpassfilteredsignal)
    else:
        stop_range=min(time_in_sec[-1], to_in_s)
        stop_range_signal=min(len(bandpassfilteredsignal), int(to_in_s/to_sec))
    
    timestamps = spikes_correct / fs
    #searched_range=(start_range, stop_range)
    range_in_s=(start_range, stop_range)
    range_in_ticks=(start_range_signal, stop_range_signal)
    spikes_in_range = spikes_correct[(spikes_correct >= range_in_ticks[0]) & (spikes_correct <= range_in_ticks[1])]
    time_in_range = time_in_sec[(time_in_sec >= range_in_s[0]) & (time_in_sec <= range_in_s[1])]
    spikes_in_range_seconds=spikes_in_range*to_sec
    
    signal_in_range=bandpassfilteredsignal[start_range_signal:(stop_range_signal+1)]
    
    
    fig, ax = plt.subplots(figsize=(20,6))
    ax = plt.plot(time_in_range, signal_in_range, c="#1E91D9")
    ax = plt.plot([time_in_range[0], time_in_range[-1]], [threshold, threshold], c="#297373")
    ax = plt.plot(spikes_in_range_seconds, [threshold-1]*spikes_in_range.shape[0], 'ro', ms=2, c="#D9580D")
    ax = plt.title('Channel %s' %channellabel)
    ax = plt.xlabel('Time in Sec, Threshold: %s' %threshold)
    ax = plt.ylabel('µ volt')
    #ax = plt.ylim(-30, 20)
    #plt.show()
    
    return fig





    
    
    
    
    