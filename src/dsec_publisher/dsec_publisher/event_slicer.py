"""
Numba-free event slicer for DSEC ``events.h5`` files.

Adapted from ``tools/dsec_preprocess.py`` (RPG DSEC), but the jitted
``get_time_indices_offsets`` helper is replaced by ``np.searchsorted`` so the
package has no ``numba`` dependency. Behaviour is identical: events are sliced
by an *exclusive* upper-bound microsecond window using the ``ms_to_idx`` map for
a conservative range and a binary search for sub-millisecond precision.
"""

import math
from typing import Dict, Optional

import h5py
import numpy as np


class EventSlicer:
    def __init__(self, h5f: h5py.File):
        self.h5f = h5f

        self.events = dict()
        for dset_str in ['p', 'x', 'y', 't']:
            self.events[dset_str] = self.h5f['events/{}'.format(dset_str)]

        # Mapping from millisecond -> event index (see DSEC docs):
        #   t[ms_to_idx[ms]]     >= ms*1000   (for ms > 0)
        #   t[ms_to_idx[ms] - 1] <  ms*1000   (for ms > 0)
        #   ms_to_idx[0] == 0
        self.ms_to_idx = np.asarray(self.h5f['ms_to_idx'], dtype='int64')

        if 't_offset' in list(h5f.keys()):
            self.t_offset = int(h5f['t_offset'][()])
        else:
            self.t_offset = 0
        self.t_final = int(self.events['t'][-1]) + self.t_offset

    def get_start_time_us(self) -> int:
        return self.t_offset

    def get_final_time_us(self) -> int:
        return self.t_final

    def ms2idx(self, time_ms: int) -> Optional[int]:
        assert time_ms >= 0
        if time_ms >= self.ms_to_idx.size:
            return None
        return int(self.ms_to_idx[time_ms])

    def get_events(self, t_start_us: int, t_end_us: int) -> Optional[Dict[str, np.ndarray]]:
        """Get events (p, x, y, t) within [t_start_us, t_end_us).

        Timestamps are absolute (i.e. they include ``t_offset``), matching the
        convention used by ``dsec_preprocess.py``. Returns ``None`` if the window
        cannot be served (out of range).
        """
        assert t_start_us < t_end_us

        # Stored timestamps are relative to t_offset.
        t_start_us -= self.t_offset
        t_end_us -= self.t_offset

        if t_end_us <= 0:
            return None

        t_start_ms = math.floor(max(t_start_us, 0) / 1000)
        t_end_ms = math.ceil(t_end_us / 1000)

        t_start_ms_idx = self.ms2idx(t_start_ms)
        # Clamp the conservative end to the last available ms bucket.
        t_end_ms_idx = self.ms2idx(t_end_ms)
        if t_end_ms_idx is None:
            t_end_ms_idx = self.events['t'].shape[0]
        if t_start_ms_idx is None:
            return None

        if t_end_ms_idx <= t_start_ms_idx:
            return None

        # Read the conservative slice once, then refine with binary search.
        time_array = np.asarray(self.events['t'][t_start_ms_idx:t_end_ms_idx])
        idx_start_offset = int(np.searchsorted(time_array, t_start_us, side='left'))
        idx_end_offset = int(np.searchsorted(time_array, t_end_us, side='left'))

        if idx_end_offset <= idx_start_offset:
            return None

        abs_start = t_start_ms_idx + idx_start_offset
        abs_end = t_start_ms_idx + idx_end_offset

        events = dict()
        events['t'] = time_array[idx_start_offset:idx_end_offset].astype(np.int64) + self.t_offset
        for dset_str in ['p', 'x', 'y']:
            events[dset_str] = np.asarray(self.events[dset_str][abs_start:abs_end])
            assert events[dset_str].size == events['t'].size
        return events
