from PySide6.QtWidgets import (
    QMainWindow, 
    QLabel, 
    QVBoxLayout, 
    QWidget, 
    QHBoxLayout, 
    QPushButton, 
    QLineEdit, 
    QComboBox, 
    QSizePolicy,
    QFrame,
    QGridLayout,
    QPlainTextEdit,
    QMessageBox,
    QMenu,
    QToolButton
)

from PySide6.QtGui import QFont, QAction
from PySide6.QtCore import QThreadPool, QTimer
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np
import pandas as pd
import os
from datetime import datetime
from datetime import timedelta

from typing import Tuple

from ..model.background_worker import BackgroundWorker
from ..model.ttl_pulse_worker import TtlPulseWorker
from ..model.stages.stage import Stage
from ..model.stages.emg_characterization_stage import EmgCharacterizationStage
from ..model.stages.mh_recruitment_curve_stage import MhRecruitmentCurveStage
from ..model.session_message import SessionMessage
from ..model.application_configuration import ApplicationConfiguration

from ..model.open_ephys_streamer import OpenEphysDataBlock, OpenEphysDataFrame

from ..model.emg_data_filter import EmgDataFilter

class MainWindow(QMainWindow):
    """
    Main application window for H-Reflex Conditioning with input fields for experimental variables
    and plots to visualize trial and live EMG data.
    """

    #region Class members / constants

    EMG_PLOTTING_SAMPLE_COUNT: int = 25000
    LIVE_EMG_WINDOW_SECONDS: float = 5.0
    EMG_PLOT_INTERVAL_MS: int = 20   # cap live EMG redraws at 50 Hz

    #endregion

    #region Constructor

    def __init__(self) -> None:
        super().__init__()

        # Initialize a variable to hold EMG signal data for plotting
        self._emg_signal_data_raw = np.zeros(MainWindow.EMG_PLOTTING_SAMPLE_COUNT)
        self._emg_signal_data_filtered = np.zeros(MainWindow.EMG_PLOTTING_SAMPLE_COUNT)
        self._emg_signal_data_abs = np.zeros(MainWindow.EMG_PLOTTING_SAMPLE_COUNT)

        # Initialize a flag to track whether a session is currently running
        self._is_session_running: bool = False
        self._is_session_paused: bool = False

        # Initialize a few flags used for plotting live emg data
        # Index 0 = plot raw/unipolar data, Index 1 = plot filtered data, Index 2 = plot abs'd data
        self._live_emg_data_plot_flags: list[bool] = [True, False, False]
        self._live_emg_data_plot_legend_names: list[str] = self._get_live_emg_legend_names()

        # Y-axis limits for the live EMG plot (µV)
        self._live_emg_y_min: float = -250.0
        self._live_emg_y_max: float = 250.0

        # Wall-clock timestamps (ms) of each stim event in the current session,
        # and the list of ScatterPlotItems (star markers) currently drawn on the live EMG plot.
        self._stim_event_times_ms: list = []
        self._live_emg_stim_markers: list = []

        # Initialize a list of stages
        self._stages: list[Stage] = []

        emg_characterization_stage: EmgCharacterizationStage = EmgCharacterizationStage()
        mh_recruitment_curve_stage: MhRecruitmentCurveStage = MhRecruitmentCurveStage()
        self._stages.append(emg_characterization_stage)
        self._stages.append(mh_recruitment_curve_stage)

        # Initialize the "selected stage"
        self._selected_stage: Stage = self._stages[0]
        self._prev_selected_stage: Stage = None

        # Initialize a variable to hold the subject name
        self._subject_name: str = ""

        # Initialize a set of messages that will be displayed in the session message box
        self._session_messages: list[SessionMessage] = []

        # Set up window title and size
        self.setWindowTitle("H-Reflex Conditioning")
        self.resize(900, 600)  # Increased the width of the window

        #Create some fonts that will be used for the ui elements
        self._regular_font: QFont = QFont("Arial", 12)
        self._bold_font: QFont = QFont("Arial", 12, QFont.Bold)
        self._large_bold_font: QFont = QFont("Arial", 20, QFont.Bold)

        # Main layout container
        self._layout: QGridLayout = QGridLayout()
        self._layout.setRowStretch(0, 0)
        self._layout.setRowStretch(1, 1)
        self._layout.setRowStretch(2, 1)
        self._layout.setRowStretch(3, 0)

        # Initialize layout sections
        self._create_top_section()
        self._create_middle_section()
        self._create_bottom_section()

        # Set the central widget for the main window layout
        central_widget = QWidget()
        central_widget.setLayout(self._layout)
        self.setCentralWidget(central_widget)

        # Initialize some debug variables
        self._frame_count: int = 0
        self._sample_count: int = 0
        self._min_sample_count: int = -1
        self._max_sample_count: int = -1
        self._frame_start = datetime.now()

        # Timestamp (ms) of the last live-EMG buffer accumulation + plot refresh.
        # Used to cap the live display redraw rate at EMG_PLOT_INTERVAL_MS regardless
        # of how fast Open Ephys sends frames (prevents UI freezing at high sample rates).
        self._last_emg_plot_ms: int = 0

        # Set the session and plot widgets on each stage
        for s in self._stages:
            s.set_session_and_trial_widgets(self._session_history_plot_widget, self._previous_trial_plot_widget)

        # Initialize the threadpool and the background worker
        self.threadpool = QThreadPool()
        self.background_worker = BackgroundWorker()
        self.background_worker.signals.data_received_signal.connect(self._on_data_received)
        self.threadpool.start(self.background_worker)

        # Periodic HTTP keepalive — fires every 4 seconds to keep the OE HTTP
        # connection warm so TTL pulse buttons never experience the ~2 s delay
        # that occurs on the first request of a freshly established connection.
        self._oe_keepalive_timer = QTimer(self)
        self._oe_keepalive_timer.setInterval(4000)
        self._oe_keepalive_timer.timeout.connect(self._send_oe_http_keepalive)
        self._oe_keepalive_timer.start()

        # Auto-feed state: toggles when the user clicks the Feed button.
        # When True, _feed_timer fires a 100 ms TTL pulse on line 3 every 1.5 min.
        self._is_feeding: bool = False
        self._feed_timer = QTimer(self)
        self._feed_timer.setInterval(ApplicationConfiguration.feed_interval_ms)
        self._feed_timer.timeout.connect(self._on_feed_timer_fired)

    #endregion

    #region Methods for creating the user interface

    def _create_top_section(self) -> None:
        """
        Creates the top section of the window, which contains UI elements
        for selecting the subject and the stage. There are also labels to
        display the booth number and information about the selected stage.
        """

        #Create the grid
        grid = QGridLayout()
        #grid.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        #Create a sub-grid on the left side for the subject and stage
        left_grid = QGridLayout()
        grid.addLayout(left_grid, 0, 0)

        #Subject label
        subject_label = QLabel("Subject: ")
        subject_label.setFont(self._bold_font) 
        left_grid.addWidget(subject_label, 0, 0)
        
        #Subject text entry
        self._subject_entry = QLineEdit("")
        self._subject_entry.setFont(self._regular_font)
        self._subject_entry.setFixedWidth(150)
        self._subject_entry.setStyleSheet("QLineEdit {color: #000000; background-color: #FFFFFF;}")

        self._subject_entry.editingFinished.connect(self._on_subject_name_edited)
        left_grid.addWidget(self._subject_entry, 0, 1)

        #Stage label
        stage_label = QLabel("Stage: ")
        stage_label.setFont(self._bold_font)
        left_grid.addWidget(stage_label, 1, 0)

        #Stage selection box
        self._stage_selection_box = QComboBox()
        self._stage_selection_box.setFixedWidth(300)
        self._stage_selection_box.setFont(self._regular_font)
        self._stage_selection_box.setStyleSheet("QComboBox {color: #000000; background-color: #FFFFFF;}")
        self._stage_selection_box.currentIndexChanged.connect(self._on_stage_selection_changed)
        left_grid.addWidget(self._stage_selection_box, 1, 1)

        #Populate the stage selection box
        stage_strings: list[str] = []
        for s in self._stages:
            stage_str: str = f"({s.stage_name}) {s.stage_description}"
            stage_strings.append(stage_str)

        self._stage_selection_box.addItems(stage_strings)

        #Protocol label
        protocol_label = QLabel("Protocol: ")
        protocol_label.setFont(self._bold_font)
        left_grid.addWidget(protocol_label, 2, 0)

        #Protocol selection combo box
        self._protocol_combo = QComboBox()
        self._protocol_combo.setFixedWidth(300)
        self._protocol_combo.setFont(self._regular_font)
        self._protocol_combo.setStyleSheet("QComboBox {color: #000000; background-color: #FFFFFF;}")
        self._protocol_combo.addItems(["PROTOCOL: OFFLINE FILTERING", "PROTOCOL: ONLINE FILTERING"])
        #Default to ONLINE (index 1) — set before connecting the signal so the handler
        #does not fire during construction when dependent widgets aren't ready yet.
        self._protocol_combo.setCurrentIndex(
            1 if ApplicationConfiguration.filtering_protocol == "ONLINE" else 0
        )
        self._protocol_combo.currentIndexChanged.connect(self._on_protocol_changed)
        left_grid.addWidget(self._protocol_combo, 2, 1)

        #Create another sub-grid on the right side that will display stage information
        right_grid = QGridLayout()
        right_grid.setColumnStretch(0, 1)
        right_grid.setColumnStretch(1, 1)
        right_grid.setColumnStretch(2, 1)
        right_grid.setColumnStretch(3, 1)

        grid.addLayout(right_grid, 0, 1)
        '''
        Currently commenting this section below out, since we are not currently using it, will add back in once we will start using it
        '''
        '''
        #Create labels for the booth name
        booth = QLabel("Booth: ")
        booth.setFont(self._bold_font)
        right_grid.addWidget(booth, 0, 0)

        self._booth_label = QLabel("NA")
        self._booth_label.setFont(self._regular_font)
        right_grid.addWidget(self._booth_label, 0, 1)

        #Create labels for the stage's VNS information
        vns = QLabel("VNS: ")
        vns.setFont(self._bold_font)
        right_grid.addWidget(vns, 0, 2)

        self._vns_label = QLabel("NA")
        self._vns_label.setFont(self._regular_font)
        right_grid.addWidget(self._vns_label, 0, 3)

        #Create labels for the stage's H-Amp information
        h_amp = QLabel("H-Amp: ")
        h_amp.setFont(self._bold_font)
        right_grid.addWidget(h_amp, 1, 0)

        self._h_amp_label = QLabel("NA")
        self._h_amp_label.setFont(self._regular_font)
        right_grid.addWidget(self._h_amp_label, 1, 1)

        #Create labels for the stage's percentile information
        percent = QLabel("Percent: ")
        percent.setFont(self._bold_font)
        right_grid.addWidget(percent, 1, 2)

        self._percent_label = QLabel("NA")
        self._percent_label.setFont(self._regular_font)
        right_grid.addWidget(self._percent_label, 1, 3)
        '''
        #Stimulation parameter panel — visible only when MH Recruitment Curve stage is selected
        self._stim_params_frame = QFrame()
        self._stim_params_frame.setFrameShape(QFrame.Shape.StyledPanel)
        sp_layout = QGridLayout(self._stim_params_frame)
        sp_layout.setContentsMargins(4, 4, 4, 4)
        sp_layout.setSpacing(4)

        sp_title = QLabel("Stim Params:")
        sp_title.setFont(self._bold_font)
        sp_layout.addWidget(sp_title, 0, 0)

        sp_layout.addWidget(QLabel("Min (mA):"), 0, 1)
        self._stim_min_amp_entry = QLineEdit(str(MhRecruitmentCurveStage.STIMULATION_AMPLITUDE_MIN))
        self._stim_min_amp_entry.setFixedWidth(55)
        sp_layout.addWidget(self._stim_min_amp_entry, 0, 2)

        sp_layout.addWidget(QLabel("Max (mA):"), 0, 3)
        self._stim_max_amp_entry = QLineEdit(str(MhRecruitmentCurveStage.STIMULATION_AMPLITUDE_MAX))
        self._stim_max_amp_entry.setFixedWidth(55)
        sp_layout.addWidget(self._stim_max_amp_entry, 0, 4)

        sp_layout.addWidget(QLabel("Step (mA):"), 1, 1)
        self._stim_step_entry = QLineEdit(str(MhRecruitmentCurveStage.STIMULATION_AMPLITUDE_STEP))
        self._stim_step_entry.setFixedWidth(55)
        sp_layout.addWidget(self._stim_step_entry, 1, 2)

        sp_layout.addWidget(QLabel("ISI (ms):"), 1, 3)
        self._stim_isi_entry = QLineEdit(str(MhRecruitmentCurveStage.MINIMUM_INTERTRIAL_INTERVAL_MILLISECONDS))
        self._stim_isi_entry.setFixedWidth(65)
        sp_layout.addWidget(self._stim_isi_entry, 1, 4)

        self._stim_sequential_checkbox = QtWidgets.QCheckBox("Sequential")
        self._stim_sequential_checkbox.setFont(self._regular_font)
        self._stim_sequential_checkbox.setChecked(False)
        sp_layout.addWidget(self._stim_sequential_checkbox, 2, 1, 1, 2)

        #Polarity toggle: "+/-" = positive-first (default), "-/+" red = negative-first
        self._stim_polarity_button = QPushButton("+/-")
        self._stim_polarity_button.setFont(self._regular_font)
        self._stim_polarity_button.setFixedWidth(45)
        self._stim_polarity_button.setToolTip("Toggle biphasic pulse polarity (positive-first / negative-first)")
        self._stim_polarity_button.clicked.connect(self._on_stim_polarity_button_clicked)
        sp_layout.addWidget(self._stim_polarity_button, 2, 3)

        self._stim_params_set_button = QPushButton("Set")
        self._stim_params_set_button.setFont(self._regular_font)
        self._stim_params_set_button.setFixedWidth(45)
        self._stim_params_set_button.clicked.connect(self._on_stim_params_set_button_clicked)
        sp_layout.addWidget(self._stim_params_set_button, 2, 4)

        self._stim_params_frame.setVisible(False)
        right_grid.addWidget(self._stim_params_frame, 1, 0, 1, 4)

        #M/H wave window panel — visible only when the MH Recruitment Curve stage is selected
        self._wave_window_frame = QFrame()
        self._wave_window_frame.setFrameShape(QFrame.Shape.StyledPanel)
        ww_layout = QGridLayout(self._wave_window_frame)
        ww_layout.setContentsMargins(4, 4, 4, 4)
        ww_layout.setSpacing(4)

        m_wave_title = QLabel("M-wave:")
        m_wave_title.setFont(self._bold_font)
        ww_layout.addWidget(m_wave_title, 0, 0)
        ww_layout.addWidget(QLabel("Start (ms):"), 0, 1)
        self._m_wave_start_entry = QLineEdit("0")
        self._m_wave_start_entry.setFixedWidth(55)
        self._m_wave_start_entry.editingFinished.connect(self._on_wave_window_edited)
        ww_layout.addWidget(self._m_wave_start_entry, 0, 2)
        ww_layout.addWidget(QLabel("End (ms):"), 0, 3)
        self._m_wave_end_entry = QLineEdit("0")
        self._m_wave_end_entry.setFixedWidth(55)
        self._m_wave_end_entry.editingFinished.connect(self._on_wave_window_edited)
        ww_layout.addWidget(self._m_wave_end_entry, 0, 4)

        h_wave_title = QLabel("H-wave:")
        h_wave_title.setFont(self._bold_font)
        ww_layout.addWidget(h_wave_title, 1, 0)
        ww_layout.addWidget(QLabel("Start (ms):"), 1, 1)
        self._h_wave_start_entry = QLineEdit("0")
        self._h_wave_start_entry.setFixedWidth(55)
        self._h_wave_start_entry.editingFinished.connect(self._on_wave_window_edited)
        ww_layout.addWidget(self._h_wave_start_entry, 1, 2)
        ww_layout.addWidget(QLabel("End (ms):"), 1, 3)
        self._h_wave_end_entry = QLineEdit("0")
        self._h_wave_end_entry.setFixedWidth(55)
        self._h_wave_end_entry.editingFinished.connect(self._on_wave_window_edited)
        ww_layout.addWidget(self._h_wave_end_entry, 1, 4)

        self._wave_window_frame.setVisible(False)
        right_grid.addWidget(self._wave_window_frame, 2, 0, 1, 4)

        #Initiation threshold panel — visible only when MH Recruitment Curve stage is selected
        self._threshold_frame = QFrame()
        self._threshold_frame.setFrameShape(QFrame.Shape.StyledPanel)
        th_layout = QGridLayout(self._threshold_frame)
        th_layout.setContentsMargins(4, 4, 4, 4)
        th_layout.setSpacing(4)

        thresh_title = QLabel("Init Thresholds:")
        thresh_title.setFont(self._bold_font)
        th_layout.addWidget(thresh_title, 0, 0)

        th_layout.addWidget(QLabel("LB (µV):"), 0, 1)
        self._thresh_lb_entry = QLineEdit("0")
        self._thresh_lb_entry.setFont(self._regular_font)
        self._thresh_lb_entry.setFixedWidth(65)
        th_layout.addWidget(self._thresh_lb_entry, 0, 2)

        th_layout.addWidget(QLabel("UB (µV):"), 0, 3)
        self._thresh_ub_entry = QLineEdit("0")
        self._thresh_ub_entry.setFont(self._regular_font)
        self._thresh_ub_entry.setFixedWidth(65)
        th_layout.addWidget(self._thresh_ub_entry, 0, 4)

        self._thresh_set_button = QPushButton("Set")
        self._thresh_set_button.setFont(self._regular_font)
        self._thresh_set_button.setFixedWidth(45)
        self._thresh_set_button.clicked.connect(self._on_threshold_set_button_clicked)
        th_layout.addWidget(self._thresh_set_button, 0, 5)

        self._threshold_frame.setVisible(False)
        right_grid.addWidget(self._threshold_frame, 3, 0, 1, 4)

        #EMG Characterization init threshold panel — visible only when S1 is selected
        self._emg_char_threshold_frame = QFrame()
        self._emg_char_threshold_frame.setFrameShape(QFrame.Shape.StyledPanel)
        ec_layout = QGridLayout(self._emg_char_threshold_frame)
        ec_layout.setContentsMargins(4, 4, 4, 4)
        ec_layout.setSpacing(4)

        ec_thresh_title = QLabel("EMG Init Thresholds:")
        ec_thresh_title.setFont(self._bold_font)
        ec_layout.addWidget(ec_thresh_title, 0, 0)

        ec_layout.addWidget(QLabel("LB (µV):"), 0, 1)
        self._emg_char_thresh_lb_entry = QLineEdit(str(EmgCharacterizationStage.TRIAL_INITIATION_MIN_RANGE_MICROVOLTS))
        self._emg_char_thresh_lb_entry.setFont(self._regular_font)
        self._emg_char_thresh_lb_entry.setFixedWidth(65)
        ec_layout.addWidget(self._emg_char_thresh_lb_entry, 0, 2)

        ec_layout.addWidget(QLabel("UB (µV):"), 0, 3)
        self._emg_char_thresh_ub_entry = QLineEdit(str(EmgCharacterizationStage.TRIAL_INITIATION_MAX_RANGE_MICROVOLTS))
        self._emg_char_thresh_ub_entry.setFont(self._regular_font)
        self._emg_char_thresh_ub_entry.setFixedWidth(65)
        ec_layout.addWidget(self._emg_char_thresh_ub_entry, 0, 4)

        self._emg_char_thresh_set_button = QPushButton("Set")
        self._emg_char_thresh_set_button.setFont(self._regular_font)
        self._emg_char_thresh_set_button.setFixedWidth(45)
        self._emg_char_thresh_set_button.clicked.connect(self._on_emg_char_threshold_set_button_clicked)
        ec_layout.addWidget(self._emg_char_thresh_set_button, 0, 5)

        self._emg_char_threshold_frame.setVisible(True)  # S1 is the default stage
        right_grid.addWidget(self._emg_char_threshold_frame, 0, 0, 1, 4)

        #Add the primary grid to the layout object
        self._layout.addLayout(grid, 0, 0)

    def _create_middle_section(self) -> None:
        """
        Creates the middle section with two EMG data plots: one for individual trial EMG data
        and one for displaying the EMG values of the last 50 trials.
        """
        middle_grid = QGridLayout()
        middle_grid.setRowStretch(0, 0)
        middle_grid.setRowStretch(1, 1)
        middle_grid.setColumnStretch(0, 1)
        middle_grid.setColumnStretch(1, 1)
        middle_grid.setColumnStretch(2, 1)

        #Create labels for each plot
        self._session_history_plot_selection_box = QComboBox()
        self._session_history_plot_selection_box.setFont(self._regular_font)
        self._session_history_plot_selection_box.setStyleSheet("QComboBox {color: #808080; background-color: #F0F0F0;}")
        self._session_history_plot_selection_box.setEnabled(False)
        self._session_history_plot_selection_box.currentIndexChanged.connect(self._on_session_history_plot_selection_index_changed)

        if (self._selected_stage is not None):
            items: list[str] = self._selected_stage.get_session_plot_options()
            for i in items:
                self._session_history_plot_selection_box.addItem(i)

        self._session_polarity_filter_box = QComboBox()
        self._session_polarity_filter_box.setFont(self._regular_font)
        self._session_polarity_filter_box.setStyleSheet("QComboBox {color: #808080; background-color: #F0F0F0;}")
        self._session_polarity_filter_box.setEnabled(False)
        self._session_polarity_filter_box.addItem("All Pol")
        self._session_polarity_filter_box.addItem("Pol 0: Normal")
        self._session_polarity_filter_box.addItem("Pol 1: Reversed")
        self._session_polarity_filter_box.setToolTip("Filter session plots by stimulation polarity")
        self._session_polarity_filter_box.currentIndexChanged.connect(self._on_session_polarity_filter_changed)

        self._timeline_scale_box = QComboBox()
        self._timeline_scale_box.setFont(self._regular_font)
        self._timeline_scale_box.setStyleSheet("QComboBox {color: #808080; background-color: #F0F0F0;}")
        self._timeline_scale_box.setEnabled(False)
        self._timeline_scale_box.addItem("min")
        self._timeline_scale_box.addItem("hr")
        self._timeline_scale_box.setFixedWidth(55)
        self._timeline_scale_box.setToolTip("Trial Timeline x-axis time scale")
        self._timeline_scale_box.currentIndexChanged.connect(self._on_timeline_scale_changed)
        
        self._most_recent_trial_plot_selection_box = QComboBox()
        self._most_recent_trial_plot_selection_box.setFont(self._regular_font)
        self._most_recent_trial_plot_selection_box.setStyleSheet("QComboBox {color: #808080; background-color: #F0F0F0;}")
        self._most_recent_trial_plot_selection_box.setEnabled(False)
        self._most_recent_trial_plot_selection_box.currentIndexChanged.connect(self._on_most_recent_trial_plot_selection_index_changed)

        if (self._selected_stage is not None):
            items: list[str] = self._selected_stage.get_trial_plot_options()
            for i in items:
                self._most_recent_trial_plot_selection_box.addItem(i)

        self._trial_signal_tool_button: QToolButton = QToolButton()
        self._trial_signal_tool_button.setText("Choose trial signals")
        self._trial_signal_tool_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._trial_signal_tool_button.setEnabled(False)

        self._trial_signal_tool_menu: QMenu = QMenu()
        self._trial_signal_tool_button.setMenu(self._trial_signal_tool_menu)
        self._trial_signal_tool_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)

        self._update_trial_signal_tool_button()

        self.live_emg_signal_tool_button: QToolButton = QToolButton()
        self.live_emg_signal_tool_button.setText("Choose live EMG plots")

        tool_button_size_policy: QSizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.live_emg_signal_tool_button.setSizePolicy(tool_button_size_policy)

        self.live_emg_signal_tool_menu: QMenu = QMenu()
        for i in range(0, len(self._live_emg_data_plot_legend_names)):
            action: QAction = self.live_emg_signal_tool_menu.addAction(self._live_emg_data_plot_legend_names[i])
            action.setCheckable(True)
            action.setChecked(self._live_emg_data_plot_flags[i])

            action.toggled.connect(self._handle_live_emg_data_plot_selection_changed)

        self.live_emg_signal_tool_button.setMenu(self.live_emg_signal_tool_menu)
        self.live_emg_signal_tool_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)

        #Y-axis limit controls for the live EMG plot
        self._live_emg_ymin_entry = QLineEdit(str(int(self._live_emg_y_min)))
        self._live_emg_ymin_entry.setFixedWidth(55)
        self._live_emg_ymin_entry.setFont(self._regular_font)
        self._live_emg_ymax_entry = QLineEdit(str(int(self._live_emg_y_max)))
        self._live_emg_ymax_entry.setFixedWidth(55)
        self._live_emg_ymax_entry.setFont(self._regular_font)
        self._live_emg_ylim_set_button = QPushButton("Set Y")
        self._live_emg_ylim_set_button.setFixedWidth(50)
        self._live_emg_ylim_set_button.setFont(self._regular_font)
        self._live_emg_ylim_set_button.clicked.connect(self._on_live_emg_ylim_set_clicked)

        #This plot widget will show the session history
        self._session_history_plot_widget = pg.PlotWidget()
        self._session_history_plot_widget.setBackground('w')
        # Label axes: session history shows trial number vs EMG (microvolts)
        try:
            self._session_history_plot_widget.getPlotItem().setLabel('left', 'EMG (µV)')
            self._session_history_plot_widget.getPlotItem().setLabel('bottom', 'Trial #')
        except Exception:
            # Fall back if setLabel is unavailable for this PlotWidget version
            pass
        
        #This plot will show the most recent trial
        self._previous_trial_plot_widget = pg.PlotWidget()
        self._previous_trial_plot_widget.setBackground('w')
        # Label axes: most recent trial shows samples vs EMG (microvolts)
        try:
            self._previous_trial_plot_widget.getPlotItem().setLabel('left', 'EMG (µV)')
            self._previous_trial_plot_widget.getPlotItem().setLabel('bottom', 'Time (ms)')
        except Exception:
            pass
        # Y-axis lock state — read by the stage draw methods to persist the user's range
        self._previous_trial_plot_widget._trial_y_locked = False
        self._previous_trial_plot_widget._trial_y_min   = -250.0
        self._previous_trial_plot_widget._trial_y_max   = 250.0

        #This plot will show the live EMG data
        self._live_emg_graph_widget = pg.PlotWidget()
        self._initialize_live_emg_plot()
        # Label axes for live EMG (keep ranges unchanged)
        try:
            self._live_emg_graph_widget.getPlotItem().setLabel('left', 'EMG (µV)')
            self._live_emg_graph_widget.getPlotItem().setLabel('bottom', 'Time (s)')
        except Exception:
            pass
        
        # Add both plots to the middle layout
        #middle_grid.addWidget(history_plot_label, 0, 0)
        #Stack the trial plot combo box, signal chooser, and Y-limit controls in col 1
        trial_controls_layout = QHBoxLayout()
        trial_controls_layout.setContentsMargins(0, 0, 0, 0)
        trial_controls_layout.setSpacing(4)
        trial_controls_layout.addWidget(self._most_recent_trial_plot_selection_box)
        trial_controls_layout.addWidget(self._trial_signal_tool_button)

        self._trial_ymin_entry = QLineEdit("-250")
        self._trial_ymin_entry.setFixedWidth(55)
        self._trial_ymin_entry.setFont(self._regular_font)
        self._trial_ymax_entry = QLineEdit("250")
        self._trial_ymax_entry.setFixedWidth(55)
        self._trial_ymax_entry.setFont(self._regular_font)
        self._trial_ylim_set_button = QPushButton("Set Y")
        self._trial_ylim_set_button.setFixedWidth(50)
        self._trial_ylim_set_button.setFont(self._regular_font)
        self._trial_ylim_set_button.clicked.connect(self._on_trial_ylim_set_clicked)

        trial_controls_layout.addWidget(QLabel("Y min:"))
        trial_controls_layout.addWidget(self._trial_ymin_entry)
        trial_controls_layout.addWidget(QLabel("Y max:"))
        trial_controls_layout.addWidget(self._trial_ymax_entry)
        trial_controls_layout.addWidget(self._trial_ylim_set_button)

        #Pack the live EMG button and Y-limit controls together in one row
        live_emg_controls_layout = QHBoxLayout()
        live_emg_controls_layout.setContentsMargins(0, 0, 0, 0)
        live_emg_controls_layout.setSpacing(4)
        live_emg_controls_layout.addWidget(self.live_emg_signal_tool_button)
        live_emg_controls_layout.addWidget(QLabel("Y min:"))
        live_emg_controls_layout.addWidget(self._live_emg_ymin_entry)
        live_emg_controls_layout.addWidget(QLabel("Y max:"))
        live_emg_controls_layout.addWidget(self._live_emg_ymax_entry)
        live_emg_controls_layout.addWidget(self._live_emg_ylim_set_button)

        session_plot_controls_layout = QHBoxLayout()
        session_plot_controls_layout.setContentsMargins(0, 0, 0, 0)
        session_plot_controls_layout.setSpacing(4)
        session_plot_controls_layout.addWidget(self._session_history_plot_selection_box)
        session_plot_controls_layout.addWidget(self._session_polarity_filter_box)
        session_plot_controls_layout.addWidget(self._timeline_scale_box)
        middle_grid.addLayout(session_plot_controls_layout, 0, 0)
        middle_grid.addLayout(trial_controls_layout, 0, 1)
        middle_grid.addLayout(live_emg_controls_layout, 0, 2)

        middle_grid.addWidget(self._session_history_plot_widget, 1, 0)
        middle_grid.addWidget(self._previous_trial_plot_widget, 1, 1)
        middle_grid.addWidget(self._live_emg_graph_widget, 1, 2)

        #Add this section to the window's layout
        self._layout.addLayout(middle_grid, 1, 0)

    def _create_bottom_section(self) -> None:
        """
        Creates the bottom section with a message box and buttons for the user to start/stop the session
        and also to deliver feeds to the animal.
        """

        #Create the bottom layout as a grid
        bottom_layout: QGridLayout = QGridLayout()
        bottom_layout.setColumnStretch(0, 1)
        bottom_layout.setColumnStretch(1, 0)
        bottom_layout.setRowStretch(0, 0)
        bottom_layout.setRowStretch(1, 1)

        #Create a grid for the message box and command box
        message_command_layout: QGridLayout = QGridLayout()
        message_command_layout.setRowStretch(0, 1)
        message_command_layout.setRowStretch(1, 0)

        #Create a label for the session message box
        session_message_box_label = QLabel("Session messages")
        session_message_box_label.setFont(self._bold_font)
        session_message_box_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

        #Command entry
        self._command_entry = QLineEdit("")
        self._command_entry.setFont(self._regular_font)
        self._command_entry.setPlaceholderText("Enter a command...")
        self._command_entry.setStyleSheet("QLineEdit {color: #000000; background-color: #FFFFFF;}")
        self._command_entry.returnPressed.connect(self._on_user_command_entered)
        
        #Create a session message box
        self._session_message_box = QPlainTextEdit()
        self._session_message_box.setFont(self._regular_font)
        self._session_message_box.setReadOnly(True)

        #Create 4 buttons: start/stop, pause, feed, and stim
    
        self._start_stop_button = QPushButton("Start")
        self._start_stop_button.setFont(self._large_bold_font)
        self._start_stop_button.setFixedWidth(200)
        self._start_stop_button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Expanding)
        self._start_stop_button.setStyleSheet('QPushButton {color: #9D9D9D;}')
        self._start_stop_button.setEnabled(False)
        self._start_stop_button.clicked.connect(self._on_start_stop_button_clicked)

        self._pause_button = QPushButton("Pause")
        self._pause_button.setFont(self._large_bold_font)
        self._pause_button.setFixedWidth(200)
        self._pause_button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Expanding)
        self._pause_button.setEnabled(False)
        self._pause_button.clicked.connect(self._on_pause_button_clicked)

        self._feed_button = QPushButton("Feed")
        self._feed_button.setFont(self._large_bold_font)
        self._feed_button.setFixedWidth(200)
        self._feed_button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Expanding)
        self._feed_button.setEnabled(False)
        self._feed_button.clicked.connect(self._on_feed_button_clicked)

        #Stim button (UI-only stub for now)
        self._stim_button = QPushButton("Stim")
        self._stim_button.setFont(self._large_bold_font)
        self._stim_button.setFixedWidth(200)
        self._stim_button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Expanding)
        self._stim_button.setEnabled(False)
        self._stim_button.clicked.connect(self._on_stim_button_clicked)

        #Create a grid layout to hold the buttons
        button_layout = QGridLayout()
        button_layout.setRowStretch(0, 1)
        button_layout.setRowStretch(1, 1)
        button_layout.setRowStretch(2, 1)
        button_layout.setRowStretch(3, 1)

        #Add the buttons to the button layout
        button_layout.addWidget(self._start_stop_button, 0, 0)
        button_layout.addWidget(self._pause_button, 1, 0)
        button_layout.addWidget(self._feed_button, 2, 0)
        button_layout.addWidget(self._stim_button, 3, 0)

        # ── TTL Output (OE HTTP) section ──────────────────────────────────

        ttl_separator = QFrame()
        ttl_separator.setFrameShape(QFrame.Shape.HLine)
        ttl_separator.setFrameShadow(QFrame.Shadow.Sunken)
        button_layout.addWidget(ttl_separator, 4, 0)

        ttl_section_label = QLabel("TTL Output (OE HTTP)")
        ttl_section_label.setFont(self._bold_font)
        button_layout.addWidget(ttl_section_label, 5, 0)

        # Host + Port + Test row
        oe_addr_row = QHBoxLayout()
        oe_addr_row.setSpacing(4)
        oe_addr_row.addWidget(QLabel("Host:"))
        self._oe_hostname_entry = QLineEdit(ApplicationConfiguration.oe_http_hostname)
        self._oe_hostname_entry.setFixedWidth(90)
        self._oe_hostname_entry.setFont(self._regular_font)
        oe_addr_row.addWidget(self._oe_hostname_entry)
        oe_addr_row.addWidget(QLabel("Port:"))
        self._oe_port_entry = QLineEdit(str(ApplicationConfiguration.oe_http_port))
        self._oe_port_entry.setFixedWidth(50)
        self._oe_port_entry.setFont(self._regular_font)
        oe_addr_row.addWidget(self._oe_port_entry)
        self._oe_test_button = QPushButton("Test")
        self._oe_test_button.setFont(self._regular_font)
        self._oe_test_button.setFixedWidth(45)
        self._oe_test_button.setToolTip("Ping the Open Ephys HTTP server to verify it is reachable")
        self._oe_test_button.clicked.connect(self._on_oe_test_button_clicked)
        oe_addr_row.addWidget(self._oe_test_button)
        self._oe_status_label = QLabel("●")
        self._oe_status_label.setStyleSheet("color: gray; font-size: 18px;")
        oe_addr_row.addWidget(self._oe_status_label)
        oe_addr_row.addStretch()
        button_layout.addLayout(oe_addr_row, 6, 0)

        # TTL pulse row
        ttl1_row = QHBoxLayout()
        ttl1_row.setSpacing(4)
        ttl1_row.addWidget(QLabel("Line:"))
        self._ne_line1_entry = QLineEdit(str(ApplicationConfiguration.ttl_line_1))
        self._ne_line1_entry.setFixedWidth(35)
        self._ne_line1_entry.setFont(self._regular_font)
        ttl1_row.addWidget(self._ne_line1_entry)
        self._ttl1_pulse_button = QPushButton("Pulse")
        self._ttl1_pulse_button.setFont(self._regular_font)
        self._ttl1_pulse_button.setToolTip("Send a hardware 5V TTL pulse via Open Ephys HTTP API")
        self._ttl1_pulse_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._ttl1_pulse_button.clicked.connect(self._on_ttl_pulse_button_clicked)
        ttl1_row.addWidget(self._ttl1_pulse_button, 1)
        button_layout.addLayout(ttl1_row, 7, 0)

        # Pulse duration row
        pulse_dur_row = QHBoxLayout()
        pulse_dur_row.setSpacing(4)
        pulse_dur_row.addWidget(QLabel("Pulse dur (ms):"))
        self._ne_pulse_duration_entry = QLineEdit("100")
        self._ne_pulse_duration_entry.setFixedWidth(55)
        self._ne_pulse_duration_entry.setFont(self._regular_font)
        self._ne_pulse_duration_entry.setToolTip(
            "Duration (ms) of the 5V hardware TTL pulse sent via ACQBOARD TRIGGER"
        )
        pulse_dur_row.addWidget(self._ne_pulse_duration_entry)
        pulse_dur_row.addStretch()
        button_layout.addLayout(pulse_dur_row, 8, 0)

        #Add the message box and command box to the message command layout
        message_command_layout.addWidget(self._session_message_box, 0, 0)
        message_command_layout.addWidget(self._command_entry, 1, 0)
        
        #Add the session message box to the bottom layout
        bottom_layout.addWidget(session_message_box_label, 0, 0)
        bottom_layout.addLayout(message_command_layout, 1, 0)

        #Add the button layout to the bottom layout
        bottom_layout.addLayout(button_layout, 0, 1, 2, 1)

        #Add the bottom layout to the primary grid layout
        self._layout.addLayout(bottom_layout, 2, 0)

        pass

    #endregion

    #region Overrides

    def closeEvent(self, event):
        '''
        This method is called when the user attempts to close the application window.
        This method handles gracefully shutting the application down.
        '''

        #Disconnect from the data received signal
        self.background_worker.signals.data_received_signal.disconnect(self._on_data_received)

        #Shut down the background thread
        self.background_worker.cancel()

        #Close the connection to the AM Systems Model 4100 if it exists
        ApplicationConfiguration.disconnect_from_am_systems_4100()

        #Disconnect from the Network Events plugin if connected
        ApplicationConfiguration.disconnect_from_network_events()

        #Accept the event
        event.accept()

    #endregion

    #region Event handlers

    def _on_subject_name_edited (self) -> None:
        '''
        This method handles input of the subject name by the user. It filters
        the subject name so that it fits the criteria for subject names, and then
        sets the text on the subject entry field to be the validated text.
        '''

        #Get the text entered by the user
        current_text: str = self._subject_entry.text()

        #Filter the string so it only contains the following allowed characters:
        #   1. Alphanumeric characters
        #   2. Hyphens or underscores
        #All other character, including whitespace characters, will be removed.
        current_text = ''.join([
            c if (c.isalnum() or c == '-' or c == '_') else '' for c in current_text
        ])

        #Now change the string to all uppercase characters, and set the subject name private variable.
        self._subject_name = current_text.upper()

        #Now set the text on the text entry to the updated subject name
        self._subject_entry.setText(self._subject_name)

        #Check to see if the start/stop button should be enabled
        if (len(self._subject_entry.text()) > 0) and (self._selected_stage is not None):
            #If so...

            #Enable the start/stop button
            self._start_stop_button.setEnabled(True)
            self._start_stop_button.setStyleSheet('QPushButton {color: green;}')
        else:
            #If not...

            #Disable the start/stop button
            self._start_stop_button.setEnabled(False)
            self._start_stop_button.setStyleSheet('QPushButton {color: #9D9D9D;}')

    def _on_stage_selection_changed (self) -> None:
        '''
        This function is executed anytime the user selects a stage
        in the stage selection box.
        '''

        #Set the selected stage
        current_stage_index = self._stage_selection_box.currentIndex()
        if (current_stage_index >= 0):
            self._selected_stage = self._stages[current_stage_index]
        else:
            self._selected_stage = None
        
        #Re-populate the session plot selection combo box and the trial plot selection combo box
        if (self._selected_stage is not None):
            items: list[str] = self._selected_stage.get_session_plot_options()

            if (hasattr(self, "_session_history_plot_selection_box")) and (self._session_history_plot_selection_box is not None):
                self._session_history_plot_selection_box.clear()
                for i in items:
                    self._session_history_plot_selection_box.addItem(i)

            items: list[str] = self._selected_stage.get_trial_plot_options()

            if (hasattr(self, "_most_recent_trial_plot_selection_box")) and (self._most_recent_trial_plot_selection_box is not None):
                self._most_recent_trial_plot_selection_box.clear()
                for i in items:
                    self._most_recent_trial_plot_selection_box.addItem(i)

        #Rebuild the trial signal chooser menu for the newly selected stage
        if (hasattr(self, "_trial_signal_tool_button")) and (self._trial_signal_tool_button is not None):
            self._update_trial_signal_tool_button()

        #Show/hide the M/H wave window panel and threshold panel based on stage type
        is_recruitment_curve: bool = (
            self._selected_stage is not None and
            self._selected_stage.stage_type == Stage.STAGE_TYPE_RECRUITMENT_CURVE
        )
        is_emg_char: bool = (
            self._selected_stage is not None and
            self._selected_stage.stage_type == Stage.STAGE_TYPE_EMG_CHARACTERIZATION
        )
        if hasattr(self, "_stim_params_frame") and self._stim_params_frame is not None:
            self._stim_params_frame.setVisible(is_recruitment_curve)
        if hasattr(self, "_wave_window_frame") and self._wave_window_frame is not None:
            self._wave_window_frame.setVisible(is_recruitment_curve)
        if hasattr(self, "_threshold_frame") and self._threshold_frame is not None:
            self._threshold_frame.setVisible(is_recruitment_curve)
        if hasattr(self, "_emg_char_threshold_frame") and self._emg_char_threshold_frame is not None:
            self._emg_char_threshold_frame.setVisible(is_emg_char)

        #Manage stage signal connection so pre-session commands give visible feedback.
        #Disconnect from the previous stage (if any), connect to the newly selected stage.
        if not self._is_session_running:
            if self._prev_selected_stage is not None:
                try:
                    self._prev_selected_stage.signals.new_message.disconnect(self._on_message_received_from_stage)
                except RuntimeError:
                    pass
            if self._selected_stage is not None:
                self._selected_stage.signals.new_message.connect(self._on_message_received_from_stage)
            self._prev_selected_stage = self._selected_stage

        #Check to see if the start/stop button should be enabled
        if (len(self._subject_entry.text()) > 0) and (self._selected_stage is not None):
            #If so...

            #Enable the start/stop button
            if (hasattr(self, "_start_stop_button")) and (self._start_stop_button is not None):
                self._start_stop_button.setEnabled(True)
                self._start_stop_button.setStyleSheet('QPushButton {color: green;}')
        else:
            #If not...

            #Disable the start/stop button
            if (hasattr(self, "_start_stop_button")) and (self._start_stop_button is not None):
                self._start_stop_button.setEnabled(False)
                self._start_stop_button.setStyleSheet('QPushButton {color: #9D9D9D;}')

    def _on_data_received (self, received: OpenEphysDataFrame) -> None:
        #Update the global sample rate from the incoming frame metadata (before filter init)
        if received.channel_data_blocks and received.channel_data_blocks[0].sample_rate > 0:
            ApplicationConfiguration.sample_rate = received.channel_data_blocks[0].sample_rate

        #Initialize the filter if necessary (OFFLINE mode only — ONLINE receives pre-filtered data)
        if ApplicationConfiguration.filtering_protocol == "OFFLINE":
            if (EmgDataFilter.sos is None):
                EmgDataFilter.initialize_filter()

        #Calculate the uninitialized fields in the dataframe
        received.calculate_fields()

        #Grab the data was sent from Open Ephys
        data = received.diff_data_block
        sample_rate = received.channel_data_blocks[0].sample_rate

        #Accumulate live EMG display buffers in-place (zero allocations).
        #Always runs every frame so the scrolling trace is continuous.
        _n: int = MainWindow.EMG_PLOTTING_SAMPLE_COUNT
        _n_new: int = min(len(received.diff_data_block), _n)
        _keep: int = _n - _n_new
        self._emg_signal_data_raw[:_keep]      = self._emg_signal_data_raw[_n_new:_n]
        self._emg_signal_data_raw[_keep:]       = received.diff_data_block[-_n_new:]
        self._emg_signal_data_filtered[:_keep] = self._emg_signal_data_filtered[_n_new:_n]
        self._emg_signal_data_filtered[_keep:]  = received.filtered_data_block[-_n_new:]
        self._emg_signal_data_abs[:_keep]      = self._emg_signal_data_abs[_n_new:_n]
        self._emg_signal_data_abs[_keep:]       = received.abs_data_block[-_n_new:]

        #Gate the actual plot redraw at EMG_PLOT_INTERVAL_MS (50 Hz cap).
        now_ms: int = int(datetime.now().timestamp() * 1000)
        do_plot_emg: bool = (now_ms - self._last_emg_plot_ms >= MainWindow.EMG_PLOT_INTERVAL_MS)
        if do_plot_emg:
            self._last_emg_plot_ms = now_ms
        
        #For debugging purposes, keep a count of how many frames per second we are achieving
        self._frame_count += 1
        self._sample_count += len(data)

        if (self._min_sample_count == -1) or (len(data) < self._min_sample_count):
            self._min_sample_count = len(data)
        
        if (self._max_sample_count == -1) or (len(data) > self._max_sample_count):
            self._max_sample_count = len(data)
        
        current_time = datetime.now()
        if (current_time >= (self._frame_start + timedelta(seconds=1))):
            samples_per_frame = self._sample_count / self._frame_count
            print(f"Frame count = {self._frame_count}, Sample count = {self._sample_count}, Samples per frame = {samples_per_frame}, Min samples per frame = {self._min_sample_count}, Max samples per frame = {self._max_sample_count}, Sample rate = {sample_rate}")
            self._min_sample_count = -1
            self._max_sample_count = -1
            self._frame_start = current_time
            self._frame_count = 0
            self._sample_count = 0
        
        #Check to see if a session is actively running
        if (self._is_session_running) and (not (self._is_session_paused)):
            #If so, process the data through the selected stage
            self._selected_stage.process(received)

        #Plot the live emg data (rate-limited to EMG_PLOT_INTERVAL_MS)
        if do_plot_emg:
            self._plot_live_emg()

        #Return from this function
        return

    def _on_start_stop_button_clicked (self) -> None:
        if (not self._is_session_running):
            #Clear the list of messages
            self._clear_session_messages()

            #Signal already connected from stage selection; no additional connect needed here.

            #Initialize the selected stage
            init_result: tuple[bool, str] = self._selected_stage.initialize(self._subject_name)

            #Check to see if the stage can proceed
            if (not init_result[0]):
                #No disconnect needed — signal remains connected for pre-session command feedback.

                #If not, then display an error dialog box to the user
                error_message: str = init_result[1]

                dlg: QMessageBox = QMessageBox(self)
                dlg.setWindowTitle("Error during stage initialization")
                dlg.setText(error_message)
                dlg.exec()
                
                #Return immediately from this function
                return

            #Clear stim events from any previous session and connect the stim overlay signal
            self._stim_event_times_ms = []
            try:
                self._selected_stage.signals.stim_triggered.connect(self._on_stim_triggered)
            except Exception:
                pass

            #Add a session message indicating the session is beginning
            message: SessionMessage = SessionMessage(f"Session started ({self._subject_name})")
            self._session_messages.append(message)
            self._update_session_messages()

            #Set the "session running" flag to True
            self._is_session_running = True

            #Set the text and text color on the start/stop button
            self._start_stop_button.setText("Stop")
            self._start_stop_button.setStyleSheet('QPushButton {color: red;}')

            #Disable the subject entry, stage selection box, and protocol combo
            self._subject_entry.setEnabled(False)
            self._subject_entry.setStyleSheet("QLineEdit {color: #808080; background-color: #F0F0F0;}")

            self._stage_selection_box.setEnabled(False)
            self._stage_selection_box.setStyleSheet("QComboBox {color: #808080; background-color: #F0F0F0;}")

            self._protocol_combo.setEnabled(False)
            self._protocol_combo.setStyleSheet("QComboBox {color: #808080; background-color: #F0F0F0;}")

            #Enable the plot selection combo boxes
            self._session_history_plot_selection_box.setEnabled(True)
            self._session_history_plot_selection_box.setStyleSheet("QComboBox {color: #000000; background-color: #FFFFFF;}")
            self._session_polarity_filter_box.setEnabled(True)
            self._session_polarity_filter_box.setStyleSheet("QComboBox {color: #000000; background-color: #FFFFFF;}")
            self._timeline_scale_box.setEnabled(True)
            self._timeline_scale_box.setStyleSheet("QComboBox {color: #000000; background-color: #FFFFFF;}")
            self._most_recent_trial_plot_selection_box.setEnabled(True)
            self._most_recent_trial_plot_selection_box.setStyleSheet("QComboBox {color: #000000; background-color: #FFFFFF;}")

            #Enable the trial signal chooser if the stage exposes signal options
            if self._selected_stage.get_trial_signal_options():
                self._trial_signal_tool_button.setEnabled(True)

            #Enable the pause and feed buttons
            self._pause_button.setEnabled(True)
            self._feed_button.setEnabled(True)
            #Enable the stim button (UI stub)
            if hasattr(self, "_stim_button"):
                self._stim_button.setEnabled(True)
        else:
            #Signal remains connected after session stop so pre-session commands (e.g. thresh)
            #still show feedback without requiring a new stage selection.

            #Set the "session running" flag to False
            self._is_session_running = False

            #Disconnect the stim overlay signal and clear any remaining stim markers
            try:
                self._selected_stage.signals.stim_triggered.disconnect(self._on_stim_triggered)
            except Exception:
                pass
            self._stim_event_times_ms = []
            for marker in self._live_emg_stim_markers:
                try:
                    self._live_emg_graph_widget.removeItem(marker)
                except Exception:
                    pass
            self._live_emg_stim_markers = []

            #Finalize the stage and close the data file
            self._selected_stage.finalize()

            #Update she session message box
            message: SessionMessage = SessionMessage(f"Session stopped ({self._subject_name})")
            self._session_messages.append(message)
            self._update_session_messages()

            #Check if the session was paused at the time that the user
            #pressed the "stop" button
            if (self._is_session_paused):
                #If necessary, reset the "session paused" flag
                self._is_session_paused = False

                #Also make sure the pause button has the correct text
                self._pause_button.setText("Pause")

            #Set the text and text color on the start/stop button
            #Also disable the start/stop button (it will become enabled 
            #again after the user enters a new subject name for the next experiment)
            self._start_stop_button.setText("Start")
            self._start_stop_button.setEnabled(False)
            self._start_stop_button.setStyleSheet('QPushButton {color: #9D9D9D;}')

            #Stop auto-feeding if it was active when the session was stopped
            if self._is_feeding:
                self._is_feeding = False
                self._feed_timer.stop()
                self._feed_button.setText("Feed")
                self._feed_button.setStyleSheet("")

            #Disable the pause and feed buttons
            self._pause_button.setEnabled(False)
            self._feed_button.setEnabled(False)
            #Disable the stim button
            if hasattr(self, "_stim_button"):
                self._stim_button.setEnabled(False)

            #Reset the subject name to be empty
            self._subject_entry.setText("")

            #Enable the subject entry and the stage selection box
            self._subject_entry.setEnabled(True)
            self._subject_entry.setStyleSheet("QLineEdit {color: #000000; background-color: #FFFFFF;}")

            self._stage_selection_box.setEnabled(True)
            self._stage_selection_box.setStyleSheet("QComboBox {color: #000000; background-color: #FFFFFF;}")

            self._protocol_combo.setEnabled(True)
            self._protocol_combo.setStyleSheet("QComboBox {color: #000000; background-color: #FFFFFF;}")

            #Disable the plot selection combo boxes
            self._session_history_plot_selection_box.setEnabled(False)
            self._session_history_plot_selection_box.setStyleSheet("QComboBox {color: #808080; background-color: #F0F0F0;}")
            self._session_polarity_filter_box.setCurrentIndex(0)
            self._session_polarity_filter_box.setEnabled(False)
            self._session_polarity_filter_box.setStyleSheet("QComboBox {color: #808080; background-color: #F0F0F0;}")
            self._timeline_scale_box.setCurrentIndex(0)
            self._timeline_scale_box.setEnabled(False)
            self._timeline_scale_box.setStyleSheet("QComboBox {color: #808080; background-color: #F0F0F0;}")
            self._most_recent_trial_plot_selection_box.setEnabled(False)
            self._most_recent_trial_plot_selection_box.setStyleSheet("QComboBox {color: #808080; background-color: #F0F0F0;}")
            self._trial_signal_tool_button.setEnabled(False)
    
    def _on_pause_button_clicked (self) -> None:
        #Set the "paused" flag
        self._is_session_paused = not self._is_session_paused

        #Check if the session is now paused
        if (self._is_session_paused):
            #Change the pause button text to "resume"
            self._pause_button.setText("Resume")

            #The start/stop button will remain enabled, 
            #so the user can still stop the session if they want.
            #But let's make sure the feed button is disabled
            self._feed_button.setEnabled(False)
        else:
            #Change the pause button text to "pause"
            self._pause_button.setText("Pause")

            #Enable the feed button
            self._feed_button.setEnabled(True)

    def _on_threshold_set_button_clicked(self) -> None:
        '''
        Called when the user clicks the "Set" button in the Init Thresholds panel.
        Reads LB/UB from the entry fields and forwards them directly to the
        MH Recruitment Curve stage, bypassing stage signals so feedback always
        appears in the session message box (before, during, or after a session).
        '''
        if (self._selected_stage is None or
                self._selected_stage.stage_type != Stage.STAGE_TYPE_RECRUITMENT_CURVE):
            return

        try:
            lb = float(self._thresh_lb_entry.text())
            ub = float(self._thresh_ub_entry.text())
        except ValueError:
            msg = SessionMessage("Threshold error: enter valid numbers for LB and UB.")
            self._session_messages.append(msg)
            self._update_session_messages()
            return

        result: str = self._selected_stage.set_manual_thresholds(lb, ub)
        msg = SessionMessage(result)
        self._session_messages.append(msg)
        self._update_session_messages()

    def _on_stim_params_set_button_clicked (self) -> None:
        if (self._selected_stage is None or
                self._selected_stage.stage_type != Stage.STAGE_TYPE_RECRUITMENT_CURVE):
            return
        try:
            min_amp = float(self._stim_min_amp_entry.text())
            max_amp = float(self._stim_max_amp_entry.text())
            step    = float(self._stim_step_entry.text())
            isi_ms  = int(float(self._stim_isi_entry.text()))
        except ValueError:
            msg = SessionMessage("Stim params error: enter valid numbers for Min, Max, Step, and ISI.")
            self._session_messages.append(msg)
            self._update_session_messages()
            return
        sequential = self._stim_sequential_checkbox.isChecked()
        result: str = self._selected_stage.set_stim_params(min_amp, max_amp, step, isi_ms, sequential)
        msg = SessionMessage(result)
        self._session_messages.append(msg)
        self._update_session_messages()

    def _on_stim_polarity_button_clicked (self) -> None:
        '''
        Toggles the biphasic pulse polarity on the MH Recruitment Curve stage and
        updates the button label/style to reflect the new state.
        '''
        mh_stage: MhRecruitmentCurveStage = None
        for s in self._stages:
            if s.stage_type == Stage.STAGE_TYPE_RECRUITMENT_CURVE:
                mh_stage = s
                break

        if mh_stage is None:
            return

        reversed_polarity: bool = mh_stage.toggle_polarity()

        #Reversed = red "-/+" label; normal = default "+/-"
        if reversed_polarity:
            self._stim_polarity_button.setText("-/+")
            self._stim_polarity_button.setStyleSheet("QPushButton {color: red;}")
        else:
            self._stim_polarity_button.setText("+/-")
            self._stim_polarity_button.setStyleSheet("")

    def _on_emg_char_threshold_set_button_clicked (self) -> None:
        if (self._selected_stage is None or
                self._selected_stage.stage_type != Stage.STAGE_TYPE_EMG_CHARACTERIZATION):
            return
        try:
            lb = float(self._emg_char_thresh_lb_entry.text())
            ub = float(self._emg_char_thresh_ub_entry.text())
        except ValueError:
            msg = SessionMessage("Threshold error: enter valid numbers for LB and UB.")
            self._session_messages.append(msg)
            self._update_session_messages()
            return
        result: str = self._selected_stage.set_initiation_thresholds(lb, ub)
        msg = SessionMessage(result)
        self._session_messages.append(msg)
        self._update_session_messages()

    def _on_feed_button_clicked(self) -> None:
        self._is_feeding = not self._is_feeding
        if self._is_feeding:
            self._feed_button.setText("Feed (ON)")
            self._feed_button.setStyleSheet("QPushButton {color: #007700;}")
            self._on_feed_timer_fired()   # deliver first pulse immediately
            self._feed_timer.start()
            msg = SessionMessage(
                f"Feed: auto-feed ON — {ApplicationConfiguration.feed_pulse_duration_ms} ms TTL "
                f"on line {ApplicationConfiguration.ttl_line_feed} every 1.5 min"
            )
        else:
            self._feed_button.setText("Feed")
            self._feed_button.setStyleSheet("")
            self._feed_timer.stop()
            msg = SessionMessage("Feed: auto-feed OFF")
        self._session_messages.append(msg)
        self._update_session_messages()

    def _on_feed_timer_fired(self) -> None:
        hostname = self._oe_hostname_entry.text().strip() or "localhost"
        try:
            port = int(self._oe_port_entry.text())
        except ValueError:
            port = ApplicationConfiguration.oe_http_port

        worker = TtlPulseWorker(
            ne_hostname  = ApplicationConfiguration.network_events_hostname,
            ne_port      = ApplicationConfiguration.network_events_port,
            oe_hostname  = hostname,
            oe_port      = port,
            line         = ApplicationConfiguration.ttl_line_feed,
            duration_ms  = ApplicationConfiguration.feed_pulse_duration_ms,
            http_session = ApplicationConfiguration.get_http_session(),
            button_index = 0,
        )
        worker.signals.finished.connect(self._on_feed_pulse_finished)
        self.threadpool.start(worker)

    def _on_feed_pulse_finished(self, _button_index: int, success: bool, message: str) -> None:
        prefix = "" if success else "[FAILED] "
        msg = SessionMessage(f"{prefix}Feed pulse: {message}")
        self._session_messages.append(msg)
        self._update_session_messages()

    def _on_stim_button_clicked (self) -> None:
        if (self._is_session_running) and (not self._is_session_paused):
            self._selected_stage.manual_stim()

    def _send_oe_http_keepalive(self) -> None:
        '''
        Sends a lightweight GET /api/status to the OE HTTP server in a background
        daemon thread.  Called by the periodic keepalive timer every 4 seconds so the
        shared HTTP connection stays warm and the first TTL button press after any idle
        period is never delayed.  Failures are silently ignored.
        '''
        import threading
        hostname = ApplicationConfiguration.oe_http_hostname
        port     = ApplicationConfiguration.oe_http_port

        def _ping():
            try:
                sess = ApplicationConfiguration.get_http_session()
                sess.get(f"http://{hostname}:{port}/api/status", timeout=(0.3, 2.0))
            except Exception:
                pass

        threading.Thread(target=_ping, daemon=True).start()

    def _on_oe_test_button_clicked(self) -> None:
        '''Ping the Open Ephys HTTP server at /api/status to verify it is reachable.'''
        hostname = self._oe_hostname_entry.text().strip() or "localhost"
        try:
            port = int(self._oe_port_entry.text())
        except ValueError:
            msg = SessionMessage("OE HTTP: invalid port number")
            self._session_messages.append(msg)
            self._update_session_messages()
            return

        ApplicationConfiguration.oe_http_hostname = hostname
        ApplicationConfiguration.oe_http_port = port

        try:
            # Use the shared persistent session so the TCP connection is
            # pre-warmed and ready before the first TTL pulse button press.
            sess = ApplicationConfiguration.get_http_session()
            r = sess.get(
                f"http://{hostname}:{port}/api/status",
                timeout=2.0
            )
            r.raise_for_status()
            self._oe_status_label.setStyleSheet("color: green; font-size: 18px;")
            msg = SessionMessage(f"OE HTTP: reachable at {hostname}:{port}")
        except Exception as exc:
            self._oe_status_label.setStyleSheet("color: red; font-size: 18px;")
            msg = SessionMessage(f"OE HTTP: unreachable — {exc}")

        self._session_messages.append(msg)
        self._update_session_messages()

    def _on_ttl_pulse_button_clicked(self) -> None:
        '''
        Send a hardware 5V TTL pulse via the Open Ephys HTTP REST API.

        Sends PUT /api/message with body {"text": "ACQBOARD TRIGGER <line> <duration_ms>"}.
        The Open Ephys Acquisition Board plugin handles this by driving the specified
        digital output line HIGH for duration_ms milliseconds, producing a real hardware
        TTL pulse visible on an oscilloscope and usable to trigger external devices.
        '''
        try:
            duration_ms = int(self._ne_pulse_duration_entry.text())
            if duration_ms <= 0:
                raise ValueError("duration must be positive")
        except ValueError:
            msg = SessionMessage("TTL Pulse: enter a positive integer duration (ms)")
            self._session_messages.append(msg)
            self._update_session_messages()
            return

        try:
            line = int(self._ne_line1_entry.text())
        except ValueError:
            line = ApplicationConfiguration.ttl_line_1

        hostname = self._oe_hostname_entry.text().strip() or "localhost"
        try:
            port = int(self._oe_port_entry.text())
        except ValueError:
            port = ApplicationConfiguration.oe_http_port

        ApplicationConfiguration.oe_http_hostname = hostname
        ApplicationConfiguration.oe_http_port = port

        self._ttl1_pulse_button.setEnabled(False)
        self._ttl1_pulse_button.setText("Pulsing…")

        worker = TtlPulseWorker(
            ne_hostname  = ApplicationConfiguration.network_events_hostname,
            ne_port      = ApplicationConfiguration.network_events_port,
            oe_hostname  = hostname,
            oe_port      = port,
            line         = line,
            duration_ms  = duration_ms,
            http_session = ApplicationConfiguration.get_http_session(),
            button_index = 1,
        )
        worker.signals.finished.connect(self._on_ttl_pulse_finished)
        self.threadpool.start(worker)

    def _on_ttl_pulse_finished(self, button_index: int, success: bool, message: str) -> None:
        '''Re-enables the TTL button and logs the outcome after the background worker completes.'''
        self._ttl1_pulse_button.setText("Pulse")
        self._ttl1_pulse_button.setEnabled(True)

        self._oe_status_label.setStyleSheet(
            "color: green; font-size: 18px;" if success else "color: red; font-size: 18px;"
        )

        prefix = "" if success else "[FAILED] "
        msg = SessionMessage(f"{prefix}TTL {button_index}: {message}")
        self._session_messages.append(msg)
        self._update_session_messages()

        # Pre-warm the connection for the next press immediately after each pulse.
        # This absorbs any OE per-connection initialization in the background so the
        # user never waits for it on the next button press.
        self._send_oe_http_keepalive()

    def _on_message_received_from_stage (self, message: SessionMessage) -> None:
        self._session_messages.append(message)
        self._update_session_messages()

    def _on_user_command_entered (self) -> None:

        #Allow commands when a session is running, OR when a stage is selected but no session
        #is running (e.g. to configure thresholds before starting a session).
        session_running_and_active: bool = (self._is_session_running) and (not (self._is_session_paused))
        pre_session_with_stage: bool = (not self._is_session_running) and (self._selected_stage is not None)

        if session_running_and_active or pre_session_with_stage:
            #Get the text that the user entered
            user_input: str = self._command_entry.text()

            #Clear the text in the UI
            self._command_entry.setText("")

            #Pass the text to the stage
            self._selected_stage.input(user_input)

        pass

    def _on_session_history_plot_selection_index_changed (self) -> None:
        if (self._is_session_running):
            current_index = self._session_history_plot_selection_box.currentIndex()
            self._selected_stage.session_plot_index = current_index

    def _on_session_polarity_filter_changed (self) -> None:
        if (self._is_session_running) and (self._selected_stage is not None):
            idx = self._session_polarity_filter_box.currentIndex()
            # index 0 = All (-1), index 1 = Pol 0, index 2 = Pol 1
            polarity_value = idx - 1
            if hasattr(self._selected_stage, 'set_session_polarity_filter'):
                self._selected_stage.set_session_polarity_filter(polarity_value)

    def _on_timeline_scale_changed (self) -> None:
        if (self._is_session_running) and (self._selected_stage is not None):
            use_hours = self._timeline_scale_box.currentIndex() == 1
            if hasattr(self._selected_stage, 'set_timeline_use_hours'):
                self._selected_stage.set_timeline_use_hours(use_hours)

    def _on_most_recent_trial_plot_selection_index_changed (self) -> None:
        if (self._is_session_running):
            current_index = self._most_recent_trial_plot_selection_box.currentIndex()
            self._selected_stage.trial_plot_index = current_index

    def _on_stim_triggered (self) -> None:
        '''Records the wall-clock time of each stimulation for the live EMG overlay.'''
        self._stim_event_times_ms.append(int(datetime.now().timestamp() * 1000))

    def _on_live_emg_ylim_set_clicked (self) -> None:
        '''Reads the Y-min/Y-max entries and applies them to the live EMG plot Y-axis.'''
        try:
            y_min = float(self._live_emg_ymin_entry.text())
            y_max = float(self._live_emg_ymax_entry.text())
        except ValueError:
            self._live_emg_ymin_entry.setText(str(int(self._live_emg_y_min)))
            self._live_emg_ymax_entry.setText(str(int(self._live_emg_y_max)))
            return

        if y_min >= y_max:
            self._live_emg_ymin_entry.setText(str(int(self._live_emg_y_min)))
            self._live_emg_ymax_entry.setText(str(int(self._live_emg_y_max)))
            return

        self._live_emg_y_min = y_min
        self._live_emg_y_max = y_max
        self._live_emg_graph_widget.setYRange(y_min, y_max, padding=0)

    def _on_trial_ylim_set_clicked(self) -> None:
        '''Reads the Y-min/Y-max entries and locks the trial/group-avg plot Y-axis.'''
        w = self._previous_trial_plot_widget
        try:
            y_min = float(self._trial_ymin_entry.text())
            y_max = float(self._trial_ymax_entry.text())
        except ValueError:
            self._trial_ymin_entry.setText(str(int(w._trial_y_min)))
            self._trial_ymax_entry.setText(str(int(w._trial_y_max)))
            return
        if y_min >= y_max:
            self._trial_ymin_entry.setText(str(int(w._trial_y_min)))
            self._trial_ymax_entry.setText(str(int(w._trial_y_max)))
            return
        w._trial_y_min    = y_min
        w._trial_y_max    = y_max
        w._trial_y_locked = True
        w.setYRange(y_min, y_max, padding=0)

    def _handle_live_emg_data_plot_selection_changed (self, checked) -> None:

        #Get the action that triggered the signal
        action: QAction = self.sender()
        if (action is not None):
            #Get the text of the action
            txt: str = action.text()

            #See which plot item the action's text matches
            for i in range(0, len(self._live_emg_data_plot_legend_names)):
                if (txt == self._live_emg_data_plot_legend_names[i]):
                    #Toggle the flag variable for this plot item
                    self._live_emg_data_plot_flags[i] = checked

                    if (i == 0):
                        self._live_emg_line_object_raw.setVisible(checked)
                    elif (i == 1):
                        self._live_emg_line_object_filtered.setVisible(checked)
                    elif (i == 2):
                        self._live_emg_line_object_abs.setVisible(checked)

                    #Break out of the loop early (no need to continue since we found the correct item already)
                    break

        pass

    #endregion

    #region Private methods

    def _update_trial_signal_tool_button (self) -> None:
        '''
        Rebuilds the "Choose trial signals" menu from the currently selected stage.
        Called on construction and whenever the stage selection changes.
        '''
        self._trial_signal_tool_menu.clear()

        if self._selected_stage is None:
            self._trial_signal_tool_button.setEnabled(False)
            return

        options: list[str] = self._selected_stage.get_trial_signal_options()
        flags: list[bool] = self._selected_stage.trial_signal_flags

        if not options:
            self._trial_signal_tool_button.setEnabled(False)
            return

        for i, option_name in enumerate(options):
            action: QAction = self._trial_signal_tool_menu.addAction(option_name)
            action.setCheckable(True)
            action.setChecked(flags[i] if i < len(flags) else False)
            action.toggled.connect(self._handle_trial_signal_selection_changed)

        #Enable only when a session is running; will be toggled by start/stop logic
        self._trial_signal_tool_button.setEnabled(self._is_session_running)

    def _handle_trial_signal_selection_changed (self, checked: bool) -> None:
        action: QAction = self.sender()
        if action is None or self._selected_stage is None:
            return

        txt: str = action.text()
        options: list[str] = self._selected_stage.get_trial_signal_options()
        new_flags: list[bool] = list(self._selected_stage.trial_signal_flags)

        for i, option_name in enumerate(options):
            if txt == option_name and i < len(new_flags):
                new_flags[i] = checked
                break

        self._selected_stage.trial_signal_flags = new_flags

    def _on_protocol_changed (self) -> None:
        '''
        Called when the user changes the filtering protocol combo box.
        Updates ApplicationConfiguration, resets the filter, and rebuilds
        the live EMG plot and tool menu with protocol-appropriate legend names.
        '''
        protocol_text: str = self._protocol_combo.currentText()
        ApplicationConfiguration.filtering_protocol = "ONLINE" if "ONLINE" in protocol_text else "OFFLINE"

        #Reset the filter states so they will be re-initialized on the next OFFLINE data frame
        EmgDataFilter.sos = None
        EmgDataFilter.filter_state = None
        EmgDataFilter.filter_state_unipolar = None

        #Rebuild the live EMG plot with updated legend names
        self._live_emg_data_plot_legend_names = self._get_live_emg_legend_names()
        self._initialize_live_emg_plot()

        #Rebuild the live EMG signal tool menu to reflect new legend names
        self.live_emg_signal_tool_menu.clear()
        for i in range(0, len(self._live_emg_data_plot_legend_names)):
            action: QAction = self.live_emg_signal_tool_menu.addAction(self._live_emg_data_plot_legend_names[i])
            action.setCheckable(True)
            action.setChecked(self._live_emg_data_plot_flags[i])
            action.toggled.connect(self._handle_live_emg_data_plot_selection_changed)

    def _on_wave_window_edited (self) -> None:
        '''
        Called when any of the M/H wave window QLineEdit fields finish editing.
        Parses the values and applies them to the MH Recruitment Curve stage.
        If a field cannot be parsed as a float, it is reset to the current stage value.
        '''
        #Find the MH Recruitment Curve stage
        mh_stage: MhRecruitmentCurveStage = None
        for s in self._stages:
            if s.stage_type == Stage.STAGE_TYPE_RECRUITMENT_CURVE:
                mh_stage = s
                break

        if mh_stage is None:
            return

        def parse_and_update(entry: QLineEdit, current_val: float) -> float:
            try:
                return float(entry.text())
            except ValueError:
                entry.setText(str(current_val))
                return current_val

        mh_stage._user_m_wave_window_start_time_milliseconds = parse_and_update(
            self._m_wave_start_entry, mh_stage._user_m_wave_window_start_time_milliseconds)
        mh_stage._user_m_wave_window_end_time_milliseconds = parse_and_update(
            self._m_wave_end_entry, mh_stage._user_m_wave_window_end_time_milliseconds)
        mh_stage._user_h_wave_window_start_time_milliseconds = parse_and_update(
            self._h_wave_start_entry, mh_stage._user_h_wave_window_start_time_milliseconds)
        mh_stage._user_h_wave_window_end_time_milliseconds = parse_and_update(
            self._h_wave_end_entry, mh_stage._user_h_wave_window_end_time_milliseconds)

    def _get_live_emg_legend_names (self) -> list[str]:
        '''Returns legend names for the live EMG plot appropriate to the current filtering protocol.'''
        if ApplicationConfiguration.filtering_protocol == "ONLINE":
            return ["Unipolar EMG (ch0)", "Filtered EMG data", "Abs-value EMG data"]
        else:
            return ["Raw EMG data", "Filtered EMG data", "Abs-value EMG data"]

    def _update_session_messages (self) -> None:
        self._session_message_box.appendHtml(self._session_messages[-1].formatted_message_text)

        pass

    def _clear_session_messages (self) -> None:
        #Clear the session messages
        self._session_messages.clear()

        #Clear the UI edit box
        self._session_message_box.clear()

    #endregion

    #region Plot Methods

    def _initialize_live_emg_plot (self) -> None:
        '''
        Configures the Live EMG plot. Safe to call multiple times (e.g. on protocol change);
        clears existing items before re-adding them.
        '''

        #Clear any previously added plot items (and their legend entries)
        self._live_emg_graph_widget.clear()

        #Any stim markers from previous sessions were removed by clear(); reset the list
        self._live_emg_stim_markers = []

        #Style the plot
        self._live_emg_graph_widget.setBackground('w')
        self._live_emg_graph_widget.setYRange(self._live_emg_y_min, self._live_emg_y_max, padding=0)
        self._live_emg_graph_widget.setXRange(-MainWindow.LIVE_EMG_WINDOW_SECONDS, 0, padding=0)
        self._live_emg_graph_widget.getPlotItem().setLabel('bottom', 'Time (s)')
        self._live_emg_graph_widget.getPlotItem().getAxis('bottom').setPen(pg.mkPen(color='k', width=2))
        self._live_emg_graph_widget.addLegend(offset=(0, 0))

        #Create the line object that will be used for updating the data
        pen_0 = pg.mkPen(color = (0, 0, 255), width = 2)
        pen_1 = pg.mkPen(color = (255, 0, 0), width = 2)
        pen_2 = pg.mkPen(color = (0, 255, 0), width = 2)
        #self._live_emg_x_data = list(range(0, len(self._emg_signal_data_raw)))
        self._live_emg_x_data = [(i - len(self._emg_signal_data_raw)) / ApplicationConfiguration.sample_rate for i in range(0, len(self._emg_signal_data_raw))]
        self._live_emg_line_object_raw = self._live_emg_graph_widget.plot(self._live_emg_x_data, self._emg_signal_data_raw, pen = pen_0, name=self._live_emg_data_plot_legend_names[0])
        self._live_emg_line_object_filtered = self._live_emg_graph_widget.plot(self._live_emg_x_data, self._emg_signal_data_filtered, pen = pen_1, name=self._live_emg_data_plot_legend_names[1])
        self._live_emg_line_object_abs = self._live_emg_graph_widget.plot(self._live_emg_x_data, self._emg_signal_data_abs, pen = pen_2, name=self._live_emg_data_plot_legend_names[2])

        self._live_emg_line_object_raw.setVisible(self._live_emg_data_plot_flags[0])
        self._live_emg_line_object_filtered.setVisible(self._live_emg_data_plot_flags[1])
        self._live_emg_line_object_abs.setVisible(self._live_emg_data_plot_flags[2])

    def _plot_live_emg(self) -> None:
        """
        Plots the current live EMG data.

        Parameters:
            figure (Figure): Matplotlib Figure object for the live EMG plot.
        """
        self._live_emg_line_object_raw.setData(self._live_emg_x_data, self._emg_signal_data_raw)
        self._live_emg_line_object_filtered.setData(self._live_emg_x_data, self._emg_signal_data_filtered)
        self._live_emg_line_object_abs.setData(self._live_emg_x_data, self._emg_signal_data_abs)

        #Update stim markers — remove markers drawn on the previous frame
        for marker in self._live_emg_stim_markers:
            try:
                self._live_emg_graph_widget.removeItem(marker)
            except Exception:
                pass
        self._live_emg_stim_markers = []

        if self._stim_event_times_ms:
            current_ms: int = int(datetime.now().timestamp() * 1000)

            #Drop events that have scrolled off the left edge
            self._stim_event_times_ms = [
                t for t in self._stim_event_times_ms
                if (t - current_ms) / 1000.0 >= -MainWindow.LIVE_EMG_WINDOW_SECONDS
            ]

            marker_y: float = self._live_emg_y_max * 0.85

            for stim_ms in self._stim_event_times_ms:
                x_pos: float = (stim_ms - current_ms) / 1000.0
                stim_marker = pg.ScatterPlotItem(
                    x=[x_pos], y=[marker_y],
                    symbol='star', size=16,
                    pen=pg.mkPen(color=(200, 0, 0), width=1.5),
                    brush=pg.mkBrush(255, 0, 0, 220)
                )
                self._live_emg_graph_widget.addItem(stim_marker)
                self._live_emg_stim_markers.append(stim_marker)

        
    #endregion
