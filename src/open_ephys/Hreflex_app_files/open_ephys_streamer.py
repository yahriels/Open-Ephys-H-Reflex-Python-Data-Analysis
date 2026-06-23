import zmq
import sys
import numpy as np
import json
import uuid
import time
import math
from typing import Tuple

from dataclasses import dataclass, field

from .emg_data_filter import EmgDataFilter
from .application_configuration import ApplicationConfiguration

OPEN_EPHYS_EXPECTED_CHANNEL_COUNT: int = 7

@dataclass
class OpenEphysDataBlock:

    timestamp: int = 0
    timestamp_received_millis: int = 0
    channel_index: int = 0
    channel_name: str = ""
    stream_name: str = ""
    num_samples: int = 0
    sample_id: int = 0
    sample_rate: float = 0
    data: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

@dataclass
class OpenEphysDigitalEvent:
    sample_num: int = 0              # Absolute OE sample counter of the event edge
    event_id: int = 0                # 1 = rising edge (ON), 0 = falling edge (OFF)
    event_channel: int = 0           # Digital channel index (0-indexed from OE)
    timestamp_received_ms: int = 0   # Wall-clock ms when this app received the event

@dataclass
class OpenEphysDataFrame:
    timestamp: int = 0
    sample_id: int = 0
    channel_data_blocks: list[OpenEphysDataBlock] = field(default_factory=lambda: [])
    timestamp_emitted: int = 0
    digital_events: list = field(default_factory=list)

    #The following data members are CALCULATED, and thus are NOT initialized with data upon
    #construction of the object
    diff_data_block: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    filtered_data_block: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    abs_data_block: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    sync_data_block: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    unipolar_filtered_data_block: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    unipolar_abs_data_block: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    feeder_adc_data_block: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    vns_adc_data_block: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    vns_waveform_data_block: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    stim_adc_data_block: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

    #region Public methods


    def calculate_fields (self) -> None:
        if ApplicationConfiguration.filtering_protocol == "ONLINE":
            #ONLINE FILTERING: channels arrive pre-processed from Open Ephys.
            #ch0 = unipolar already-filtered signal  →  diff_data_block (the "raw" display slot)
            #ch1 = bipolar diff + bandpass + abs, fully processed in Open Ephys  →  filtered_data_block / abs_data_block
            #ch2 = ADC sync line  →  sync_data_block
            if (len(self.channel_data_blocks) >= 1):
                self.diff_data_block = self.channel_data_blocks[0].data
                #In ONLINE mode ch0 is the unipolar filtered signal
                self.unipolar_filtered_data_block = self.channel_data_blocks[0].data
                self.unipolar_abs_data_block = np.abs(self.channel_data_blocks[0].data)

            if (len(self.channel_data_blocks) >= 2):
                self.filtered_data_block = self.channel_data_blocks[1].data
                self.abs_data_block = np.abs(self.channel_data_blocks[1].data)

            if (len(self.channel_data_blocks) >= 3):
                self.sync_data_block = np.abs(self.channel_data_blocks[2].data)

            if (len(self.channel_data_blocks) >= 4):
                #ch3: Tibial Stim ADC waveform (signed, no abs — preserves biphasic pulse shape)
                self.stim_adc_data_block = self.channel_data_blocks[3].data

            if (len(self.channel_data_blocks) >= 5):
                self.vns_adc_data_block = self.channel_data_blocks[4].data

            if (len(self.channel_data_blocks) >= 6):
                self.vns_waveform_data_block = self.channel_data_blocks[5].data

            if (len(self.channel_data_blocks) >= 7):
                self.feeder_adc_data_block = self.channel_data_blocks[6].data
        else:
            #OFFLINE FILTERING: perform differential subtraction and bandpass filter in-app.
            #ch0 & ch1 = raw EMG electrodes  →  diff → filter → abs
            #ch2 = ADC sync line  →  sync_data_block
            if (len(self.channel_data_blocks) >= 2):
                #Do the differential subtraction
                self.diff_data_block = self.channel_data_blocks[1].data - self.channel_data_blocks[0].data

                #Now filter the data
                self.filtered_data_block = EmgDataFilter.filter(self.diff_data_block)

                #Now take the absolute value of the data
                self.abs_data_block = np.abs(self.filtered_data_block)

                #Filter ch0 alone to produce the unipolar (non-differential) filtered signal
                self.unipolar_filtered_data_block = EmgDataFilter.filter_unipolar(self.channel_data_blocks[0].data)
                self.unipolar_abs_data_block = np.abs(self.unipolar_filtered_data_block)

            if (len(self.channel_data_blocks) >= 3):
                #Extract the ADC sync line (channel index 2) used for precise stim-onset alignment.
                #Absolute value is applied to handle any polarity ambiguity in the ADC signal.
                self.sync_data_block = np.abs(self.channel_data_blocks[2].data)

            if (len(self.channel_data_blocks) >= 4):
                #ch3: Tibial Stim ADC waveform (signed, no abs — preserves biphasic pulse shape)
                self.stim_adc_data_block = self.channel_data_blocks[3].data

            if (len(self.channel_data_blocks) >= 5):
                self.vns_adc_data_block = self.channel_data_blocks[4].data

            if (len(self.channel_data_blocks) >= 6):
                self.vns_waveform_data_block = self.channel_data_blocks[5].data

            if (len(self.channel_data_blocks) >= 7):
                self.feeder_adc_data_block = self.channel_data_blocks[6].data

    #endregion
    

    
class OpenEphysStreamer (object):

    #region Constructor

    def __init__(self, ):
        self.context = zmq.Context()
        self.data_socket = None
        self.event_socket = None
        self.poller = zmq.Poller()
        self.message_num = -1
        self.socket_waits_reply = False
        self.event_no = 0
        self.app_name = 'TxBDC H-Reflex'
        self.uuid = str(uuid.uuid4())
        self.last_heartbeat_time = 0
        self.last_reply_time = time.time()
        self.isTesting = True

    #endregion

    #region Methods

    def send_heartbeat(self):
        '''
        This method sends a heartbeat signal to the OpenEphys application
        so that it knows we still exist.
        '''

        #Compose the message
        d = {'application': self.app_name,
             'uuid': self.uuid,
             'type': 'heartbeat'}
        j_msg = json.dumps(d)

        #Send the message
        try:
            self.event_socket.send(j_msg.encode('utf-8'))
        except Exception as ex:
            print(f"An unexpected error occurred: {ex}")
            pass
        self.last_heartbeat_time = time.time()
        self.socket_waits_reply = True

    def send_event(self, event_list=None, event_type=3, sample_num=0, event_id=2, event_channel=1):
        '''
        This method composes a message for a specific event and sends the event to the OpenEphys application.
        '''

        #Check to see if we are currently waiting for a reply from OpenEphys
        if not self.socket_waits_reply:
            #If we are not waiting for a reply...

            #Check to see if there are events in the event_list to send
            self.event_no += 1
            if event_list:
                #Iterate over the event list
                for e in event_list:
                    #Send each event in the event list
                    self.send_event(event_type=e['event_type'],
                                    sample_num=e['sample_num'],
                                    event_id=e['event_id'],
                                    event_channel=e['event_channel'])
            else:
                #If there is only a single event to send...

                #Compose the event message
                de = {'type': event_type, 'sample_num': sample_num,
                      'event_id': event_id % 2 + 1,
                      'event_channel': event_channel}

                d = {'application': self.app_name,
                     'uuid': self.uuid,
                     'type': 'event',
                     'event': de}

                j_msg = json.dumps(d)

                #Send the event message
                if not self.socket_waits_reply:
                    self.event_socket.send(j_msg.encode('utf-8'), 0)
            self.socket_waits_reply = True
            self.last_reply_time = time.time()

    def initialize (self) -> None:
        if not self.data_socket:
            #Initialize the data socket and the event socket
            self.data_socket = self.context.socket(zmq.SUB)
            self.data_socket.connect("tcp://localhost:5556")

            self.event_socket = self.context.socket(zmq.REQ)
            self.event_socket.connect("tcp://localhost:5557")

            self.data_socket.setsockopt(zmq.SUBSCRIBE, b'')
            self.poller.register(self.data_socket, zmq.POLLIN)
            self.poller.register(self.event_socket, zmq.POLLIN)

    def callback(self) -> OpenEphysDataBlock:
        #Check to see if more than 2 seconds has passed since the last "heartbeat" message
        if (time.time() - self.last_heartbeat_time) > 2.:
            #Make sure we aren't currently waiting on a reply from a prior heartbeat message
            if self.socket_waits_reply:
                #We will try again...
                self.last_heartbeat_time += 1.

                #But if too much time has passed...
                if (time.time() - self.last_reply_time) > 10.:
                    # reconnecting the socket as per
                    # the "lazy pirate" pattern (see the ZeroMQ guide)
                    self.poller.unregister(self.event_socket)
                    self.event_socket.close()
                    self.event_socket = self.context.socket(zmq.REQ)
                    self.event_socket.connect("tcp://localhost:5557")
                    self.poller.register(self.event_socket)
                    self.socket_waits_reply = False
                    self.last_reply_time = time.time()
            else:
                #If we are not waiting for a reply from a prior heartbeat message, and
                #more than 2 seconds has passed since the last heartbeat message that we
                #sent, then it is time to send a new heartbeat message.
                self.send_heartbeat()

        #Get the sockets
        socks = dict(self.poller.poll(1))
        if not socks:
            return None

        #Get the data socket
        if self.data_socket in socks:
            #Retrieve a multi-part message
            try:
                message = self.data_socket.recv_multipart(zmq.NOBLOCK)
            except zmq.ZMQError as err:
                return None

            #Check to see if we have a message that was received
            if message:

                #Commenting this out since I don't think we need this.
                #if len(message) < 2:
                #    print("no frames for message: ", message[0])
                
                #Decode the message
                try:
                    header = json.loads(message[1].decode('utf-8'))
                except ValueError as e:
                    pass
                    #print("ValueError: ", e)
                    #print(message[1])
                
                #Commenting this out since I don't this we need this.
                #if self.message_num != -1 and header['message_num'] != self.message_num + 1:
                #    print("missing a message at number", self.message_num)
                
                #Get the message number
                self.message_num = header['message_num']

                #Check to see if the header type is "data"
                if header['type'] == 'data':
                    c = header['content']
                    timestamp = header['timestamp']
                    num_samples = c['num_samples']
                    channel_num = c['channel_num']
                    sample_id = c['sample_num']
                    sample_rate = c['sample_rate']
                    stream_name = c['stream']
                    channel_name = c['channel_name']

                    #Get the data from the message
                    try:
                        n_arr = np.frombuffer(message[2], dtype=np.float32)
                        n_arr = np.reshape(n_arr, num_samples)

                        #If there were samples in the data
                        if num_samples > 0:

                            #Calculate a millisecond timestamp for when our application received this data
                            ts_received: int = int(math.floor(time.time() * 1000))

                            #Package the received data into a structure that we will return to the caller
                            received_data: OpenEphysDataBlock = OpenEphysDataBlock(
                                timestamp, ts_received, channel_num, channel_name, stream_name, num_samples, sample_id, sample_rate, n_arr)

                            #Update the UI
                            return received_data

                    except IndexError as e:
                        #print(e)
                        #print(header)
                        #print(message[1])
                        #if len(message) > 2:
                        #    print(len(message[2]))
                        #else:
                        #    print("only one frame???")
                        pass

                elif header['type'] == 'event':
                    c = header.get('content', {})
                    ts_received: int = int(math.floor(time.time() * 1000))
                    return OpenEphysDigitalEvent(
                        sample_num=int(c.get('sample_num', 0)),
                        event_id=int(c.get('event_id', 0)),
                        event_channel=int(c.get('event_channel', 0)),
                        timestamp_received_ms=ts_received
                    )

                elif header['type'] == 'spike':

                    #We will not handle this message type
                    pass

                elif header['type'] == 'param':
                    c = header['content']
                    self.__dict__.update(c)
                    
                    #print(c)
                else:
                    raise ValueError("message type unknown")
            else:
                #print("got not data")
                return None
        elif self.event_socket in socks and self.socket_waits_reply:
            #If we are waiting for a reply on the event socket...

            #Retrieve any messages
            message = self.event_socket.recv()
            #print("event reply received")
            #print(message)

            #Set the socket_waits_reply flag to false
            if self.socket_waits_reply:
                self.socket_waits_reply = False

        return None

    #endregion
