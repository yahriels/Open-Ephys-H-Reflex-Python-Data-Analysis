import json
import os
from platformdirs import user_data_dir

from am_systems_4100.am_systems_4100 import AmSystems4100
from am_systems_4100.am_systems_4100 import AmSystems4100_TcpConnectionInfo

from .booth import Booth

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
    filtering_protocol: str = "OFFLINE"

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
    def set_stimulation_amplitude (amplitude_ma: float) -> None:

        stim: AmSystems4100 = ApplicationConfiguration.stimulator
        if (stim is None):
            return
        
        #Tell the stimulator unit that each phase of the biphasic pulse will be 0.8 mA
        #in amplitude.
        stim.set_event_amplitude1(int(round(amplitude_ma * 1000.0)))

    @staticmethod
    def set_biphasic_stimulus_pulse_parameters (amplitude_ma: float) -> None:
        #   Current = decided by the caller of the function
        #   Frequency = N/A
        #   Pulse phase width = 500 us
        #   Biphasic pulse
        #   Train duration = 1000 us
        #   Total pulses = 1

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
        stim.set_event_duration1(250)
        stim.set_event_period(500)

        #Tell the stimulator unit that each phase of the biphasic pulse will be 0.8 mA
        #in amplitude.
        stim.set_event_amplitude1(amplitude_ma * 1000.0)

        #Biphasic pulses do not use "duration2" and "amplitude2", so we will set them
        #to a value of 0.
        stim.set_event_duration2(0)
        stim.set_event_amplitude2(0)

        #Tell the stimulator unit that there is 0 uS interval between the two phases
        #of the biphasic pulse.
        stim.set_event_duration3(0)

        pass

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