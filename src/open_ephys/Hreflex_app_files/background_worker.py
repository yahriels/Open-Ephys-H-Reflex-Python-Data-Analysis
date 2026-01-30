from typing import Tuple
from PySide6.QtCore import QRunnable, Slot, Signal, QObject
import numpy as np
import time
import math
import bisect

from .open_ephys_streamer import OpenEphysStreamer
from .open_ephys_streamer import OpenEphysDataBlock
from .open_ephys_streamer import OpenEphysDataFrame
from .open_ephys_streamer import OPEN_EPHYS_EXPECTED_CHANNEL_COUNT

class BackgroundWorkerSignals (QObject):

    #region Signals

    data_received_signal = Signal(OpenEphysDataFrame)

    #endregion

class BackgroundWorker (QRunnable):
    '''
    Worker thread
    '''

    #region Constructor

    def __init__(self):
        super().__init__()

        #Public signals
        self.signals = BackgroundWorkerSignals()

        #Private members
        self._open_ephys_streamer = OpenEphysStreamer()
        self._should_cancel = False

    #endregion

    #region Methods

    def cancel (self):
        self._should_cancel = True

    @Slot()
    def run (self):
        '''
        This is the code executed by the background thread.
        '''

        #Initialize the open ephys streamer
        self._open_ephys_streamer.initialize()

        #Create a variable to hold a "frame" of data
        df: OpenEphysDataFrame = None

        #Iterate until the "should cancel" is set to True
        while (not self._should_cancel):

            #Process any messages to/from OpenEphys
            result: OpenEphysDataBlock = self._open_ephys_streamer.callback()

            #Check to see if any data was received
            if (result is not None) and (result.data is not None):

                #Check if a new dataframe should be created
                if (df is None):# or ((df is not None) and (result.timestamp != df.timestamp)):
                    df = OpenEphysDataFrame(result.timestamp, result.sample_id, [], 0)
                
                #Insert this data block into the dataframe's list of blocks
                #We maintain the list of blocks IN ORDER of index, so we INSERT IN ORDER
                #For this reason, we use bisect.insort rather than list.append
                #(The data blocks SHOULD come into our app already "in order", but it's not guaranteed, so 
                #it's useful to do this just to make sure)
                bisect.insort(df.channel_data_blocks, result, key=lambda x: x.channel_index)

                #Check if it is time to emit
                if (len(df.channel_data_blocks) >= OPEN_EPHYS_EXPECTED_CHANNEL_COUNT):
                    #Emit the data
                    df.timestamp_emitted = int(math.floor(time.time() * 1000))
                    self.signals.data_received_signal.emit(df)

                    #Set the dataframe to None
                    df = None

        return

    #endregion

