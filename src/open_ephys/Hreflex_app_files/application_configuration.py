import json
import os
from platformdirs import user_data_dir

from am_systems_4100.am_systems_4100 import AmSystems4100
from am_systems_4100.am_systems_4100 import AmSystems4100_TcpConnectionInfo

from .booth import Booth
from .network_events_client import OpenEphysNetworkEventsClient

class ApplicationConfiguration:

    #The name of the application
    appname: str = "H-Reflex Conditioning"

    #The author/organization of the application
    appauthor: str = "TxBDC"

    #AM Systems Model 4100 object
    stimulator: AmSystems4100 = None

    #The name of the app's configuration file
    configuration_file_name: str = "h_reflex_config.json"

    #A list of booths loaded from the configuration file
    booth_list: list[Booth] = []

    #An object to hold the current booth that is being used by the application
    current_booth: Booth = None

    #A variable that indicates whether ad-hoc booth creation is allowed
    allow_ad_hoc_booth_creation: bool = True

    #The sampling rate reported by Open Ephys (samples/second).
    #Initialized to 5000 as a safe default; updated from the first incoming data frame.
    sample_rate: float = 5000.0

    #The filtering protocol to use.
    #"OFFLINE": differential subtraction + bandpass filter performed in-app (ch0=raw, ch1=raw, ch2=sync).
    #"ONLINE":  channels arrive pre-processed from Open Ephys (ch0=unipolar filtered, ch1=bipolar+filt+abs, ch2=sync).
    filtering_protocol: str = "ONLINE"

    #Network Events plugin client for sending TTL commands to Open Ephys.
    #NOTE: The Network Events plugin uses its own independent REP socket.
    #      If you are also running the ZMQ Interface plugin (ports 5556/5557
    #      for data streaming), you MUST configure the Network Events plugin
    #      in Open Ephys to use a different port (e.g. 5558) to avoid a conflict.
    network_events_client: OpenEphysNetworkEventsClient = None
    network_events_hostname: str = "localhost"
    network_events_port: int = 59117

    #TTL output line numbers used by the two TTL pulse buttons.
    #These correspond to the digital output lines configured in Open Ephys.
    ttl_line_1: int = 1
    ttl_line_2: int = 2

    #TTL line, pulse duration, and repeat interval for the Feed button.
    ttl_line_feed: int = 3
    feed_pulse_duration_ms: int = 100
    feed_interval_ms: int = 90000  # 1.5 minutes

    #Digital input channel number (1-indexed, matching the Open Ephys UI "Digital Input N")
    #used as the primary stim-onset reference instead of the ADC analog threshold.
    #OE ZMQ events report 0-indexed event_channel; subtract 1 when comparing.
    sync_digital_channel: int = 1

    #Open Ephys HTTP REST API server address (port 37497 is the OE default).
    #Used to send ACQBOARD TRIGGER commands that generate real 5V hardware TTL pulses.
    oe_http_hostname: str = "localhost"
    oe_http_port: int = 37497

    #Persistent HTTP session — keeps the TCP connection to the OE REST API alive
    #between pulses so each TTL button press incurs no TCP handshake overhead.
    _http_session = None

    #region Configuration file methods

    def load_configuration_file () -> None:
        #Clear the booth list
        ApplicationConfiguration.booth_list.clear()

        #Get the full file path and name
        app_data_path: str = user_data_dir(ApplicationConfiguration.appname, ApplicationConfiguration.appauthor)
        file_path_and_name: str = os.path.join(app_data_path, ApplicationConfiguration.configuration_file_name)

        #Load the json file
        data: dict = None
        if (os.path.exists(file_path_and_name)):
            with open(file_path_and_name, 'r') as file:
                try:
                    data = json.load(file)
                except:
                    print("Unable to load H-Reflex application configuration file!")

        #Check to see if JSON data was loaded properly
        if (data is not None):
            #Check to see if the JOSN data includes a setting for allowing ad-hoc booth creation
            if ("allow_ad_hoc_booth_creation" in data):
                ApplicationConfiguration.allow_ad_hoc_booth_creation = data["allow_ad_hoc_booth_creation"]

            #Check to see if the JSON data includes a list of booths
            if ("booths" in data):
                booths: list = data["booths"]

                #Iterate over each booth in the JSON data
                for i in range(0, len(booths)):
                    cur_booth: dict = booths[i]
                    if ("booth_name" in cur_booth) and ("model_4100_ip_address" in cur_booth):
                        #Create a booth object

                        booth_name: str = "-1"
                        try:
                            booth_name = cur_booth["booth_name"]
                        except:
                            pass

                        ip_addr: str = ""
                        try:
                            ip_addr = cur_booth["model_4100_ip_address"]
                        except:
                            pass

                        pin_num: int = 1001
                        try:
                            pin_num = cur_booth["model_4100_pin"]
                        except:
                            pass

                        booth_obj: Booth = Booth(booth_name, ip_addr, pin_num)

                        #Add this booth to the booth list
                        ApplicationConfiguration.booth_list.append(booth_obj)

        pass

    def save_booth_to_configuration_file (booth: Booth) -> None:

        #Get the full file path and name for the configuration file
        app_data_path: str = user_data_dir(ApplicationConfiguration.appname, ApplicationConfiguration.appauthor)
        file_path_and_name: str = os.path.join(app_data_path, ApplicationConfiguration.configuration_file_name)

        #Load the configuration json file
        data: dict = None
        if (os.path.exists(file_path_and_name)):
            with open(file_path_and_name, 'r') as file:
                try:
                    data = json.load(file)
                except:
                    print("Unable to load H-Reflex application configuration file!")

        #Check to see if JSON data was loaded properly
        if (data is None):
            data = {}

        #If a list of booths does not yet exist, create it
        if ("booths" not in data):
            data["booths"] = []

        #Get the list of booths
        booths: list = data["booths"]

        #Append an item to the list of booths
        booth_dict: dict = {
            "booth_name": booth.booth_name,
            "model_4100_ip_address": booth.model_4100_ip_address,
            "model_4100_pin": booth.model_4100_pin
        }

        booths.append(booth_dict)
        
        #Check to see if the JSON data includes a flag to allow ad-hoc booth creation
        if ("allow_ad_hoc_booth_creation" not in data):
            data["allow_ad_hoc_booth_creation"] = True
        
        #Create the folder if it does not yet exist
        if (not os.path.exists(app_data_path)):
            os.makedirs(app_data_path)

        #Save the JSON file back out
        with open(file_path_and_name, 'w') as fid:
            try:
                json.dump(data, fid, indent=4)
            except Exception as e:
                print("Unable to dump json data to configuration file!")
                print(e)

        pass

    #endregion

    #region Methods

    @staticmethod
    def connect_to_am_systems_4100 () -> None:
        #Create a connection information object
        connection_info: AmSystems4100_TcpConnectionInfo = AmSystems4100_TcpConnectionInfo(
            ApplicationConfiguration.current_booth.model_4100_pin,
            ApplicationConfiguration.current_booth.model_4100_ip_address
        )

        #Connect to the stimulator
        ApplicationConfiguration.stimulator = AmSystems4100(connection_info)

    @staticmethod
    def disconnect_from_am_systems_4100 () -> None:
        if (ApplicationConfiguration.stimulator is not None):
            try:
                ApplicationConfiguration.stimulator._sock.close()
            except:
                pass
        
        ApplicationConfiguration.stimulator = None

    @staticmethod
    def set_stimulation_amplitude (amplitude_ma: float, reversed_polarity: bool = False) -> None:

        stim: AmSystems4100 = ApplicationConfiguration.stimulator
        if (stim is None):
            return

        amp_ua: int = int(round(amplitude_ma * 1000.0))
        if reversed_polarity:
            amp_ua = -amp_ua
        stim.set_event_amplitude1(amp_ua)

    @staticmethod
    def set_biphasic_stimulus_pulse_parameters (amplitude_ma: float, reversed_polarity: bool = False) -> None:
        #   Current = decided by the caller of the function
        #   Frequency = N/A
        #   Pulse phase width = 500 us
        #   Biphasic pulse
        #   Train duration = 1000 us
        #   Total pulses = 1
        #
        #   reversed_polarity=False → positive-first biphasic (normal)
        #   reversed_polarity=True  → negative-first biphasic (inverted; amplitude1 sent as negative)

        stim: AmSystems4100 = ApplicationConfiguration.stimulator
        if (stim is None):
            return

        #Stop any active stimulation
        stim.set_active(False)

        #Tell the unit to produce "current" pulses (not "voltage" pulses).
        stim.set_mode(1)

        #Tell the stimulator unit that we will provide a specific number
        #of pulses for it to generate
        stim.set_auto(1)

        #Tell the stimulator unit that there will be 0 delay between the trigger
        #and the onset of the stimulation train.
        stim.set_train_delay(0)

        #Tell the stimulator unit that we will produce 1 stimulation train.
        stim.set_train_quantity(1)

        #Tell the stimulator unit that there will be 0 delay between the onset
        #of the stimulation train and the first event within the train.
        stim.set_event_delay(0)

        #Tell the stimulator unit that we want to use biphasic pulses.
        stim.set_event_type(1)

        #Tell the stimulator unit that we will deliver exactly 1 pulse.
        stim.set_event_quantity(1)

        #Tell the stimulator unit that each phase of the biphasic pulse will be 500 uS
        #in duration.
        stim.set_event_duration1(100)
        stim.set_event_period(200)

        #Set the amplitude in microamps.  For reversed polarity, negate the value so that
        #the first phase is cathodal (negative-first) instead of anodal (positive-first).
        amp_ua: int = int(round(amplitude_ma * 1000.0))
        if reversed_polarity:
            amp_ua = -amp_ua
        stim.set_event_amplitude1(amp_ua)

        #Biphasic pulses do not use "duration2" and "amplitude2", so we will set them
        #to a value of 0.
        stim.set_event_duration2(0)
        stim.set_event_amplitude2(0)

        #Tell the stimulator unit that there is 0 uS interval between the two phases
        #of the biphasic pulse.
        stim.set_event_duration3(0)

        pass

    @staticmethod
    def get_http_session():
        '''
        Returns the shared persistent requests.Session, creating it on first call.
        The session keeps the underlying TCP connection to the OE HTTP server alive
        so that repeated ACQBOARD TRIGGER calls incur no TCP handshake penalty.
        Retries once immediately on connection errors (stale socket) with no backoff.
        '''
        if ApplicationConfiguration._http_session is None:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            s = requests.Session()
            retry = Retry(
                total=1,
                connect=1,
                read=False,
                backoff_factor=0,
                raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=2)
            s.mount('http://', adapter)
            ApplicationConfiguration._http_session = s
        return ApplicationConfiguration._http_session

    @staticmethod
    def connect_to_network_events() -> tuple[bool, str]:
        '''Create (if necessary) and connect the Network Events client.'''
        if ApplicationConfiguration.network_events_client is None:
            ApplicationConfiguration.network_events_client = OpenEphysNetworkEventsClient()
        return ApplicationConfiguration.network_events_client.connect(
            ApplicationConfiguration.network_events_hostname,
            ApplicationConfiguration.network_events_port
        )

    @staticmethod
    def disconnect_from_network_events() -> None:
        '''Disconnect and destroy the Network Events client.'''
        if ApplicationConfiguration.network_events_client is not None:
            ApplicationConfiguration.network_events_client.disconnect()
            ApplicationConfiguration.network_events_client = None

    @staticmethod
    def send_acqboard_trigger(line: int, duration_ms: int) -> tuple[bool, str]:
        '''
        Send an ACQBOARD TRIGGER command via the Open Ephys HTTP REST API.

        This broadcasts a message to all processors in the OE signal chain.
        The Open Ephys Acquisition Board plugin handles the message by driving
        the specified digital output line HIGH for duration_ms milliseconds,
        producing a real 5V hardware TTL pulse visible on an oscilloscope.

        PUT http://<hostname>:<port>/api/message
        Body: {"text": "ACQBOARD TRIGGER <line> <duration_ms>"}
        '''
        import requests
        try:
            url = (f"http://{ApplicationConfiguration.oe_http_hostname}"
                   f":{ApplicationConfiguration.oe_http_port}/api/message")
            r = requests.put(
                url,
                json={"text": f"ACQBOARD TRIGGER {line} {duration_ms}"},
                timeout=2.0
            )
            r.raise_for_status()
            return (True, r.text)
        except Exception as exc:
            return (False, str(exc))

    @staticmethod
    def set_standard_vns_stimulation_parameters () -> None:
        #Standard VNS parameters:
        #   Current = 0.8 mA (800 uA)
        #   Frequency = 30 Hz
        #   Pulse phase width = 100 us
        #   Biphasic pulse
        #   Train duration = 500 ms (500000 microseconds)
        #   Total pulses = 15
        #   Pulses are delivered every 33.333 ms (or 33333 microseconds)

        if (ApplicationConfiguration.stimulator is not None):
            ApplicationConfiguration.stimulator.set_txbdc_standard_vns_parameters()

    #endregion