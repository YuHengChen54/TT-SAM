import numpy as np



def get_peak_value(stream, pick_point=0, thresholds=None):
    """Extract peak amplitude and timing from seismic stream data."""
    data = [tr.data for tr in stream]
    data = np.array(data)
    data = data[:, pick_point:]
    vector = np.linalg.norm(data, axis=0)

    peak = max(vector)
    peak_time = np.argmax(vector, axis=0)
    peak_time += pick_point
    peak = np.log10(peak / 100)

    exceed_times = np.zeros(5)
    if thresholds is not None:
        for i, threshold in enumerate(thresholds):
            try:
                exceed_times[i] = next(
                    x for x, val in enumerate(vector) if val > threshold
                )
            except Exception as err:
                print(err)

    return peak, peak_time


def get_integrated_stream(stream):
    """Apply bandpass filtering to integrated seismic stream."""
    stream_intergrated = stream.copy()
    stream_intergrated.taper(max_percentage=0.05, type="cosine")
    stream_intergrated.integrate()
    stream_intergrated.filter("bandpass", freqmin=0.075, freqmax=10)
    return stream_intergrated


def get_integrated_stream_second(stream):
    """Apply highpass filtering to integrated seismic stream."""
    stream_intergrated = stream.copy()
    stream_intergrated.integrate()
    stream_intergrated.filter("highpass", freq=0.075)
    return stream_intergrated
