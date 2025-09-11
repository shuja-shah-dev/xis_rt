import os


class AppConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppConfig, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.camera_config = None
        self.model_selection = None
        self.video_path = None
        self.cti_file_location = None
        self.input_type = None
        self.video_on = ""
        self.measurementMode = ""

    def set_measurement_mode(self, mode):
        self.measurementMode = mode
    
    def get_measurement_mode(self):
        return self.measurementMode

    def set_camera_config(self, cti_file_location):
        self.input_type = "camera"
        self.cti_file_location = cti_file_location
        self.video_path = None

    def set_video_config(self, src):
        self.input_type = "video"
        if src == "baked_baguette":
            self.video_path = os.path.join(
                os.path.dirname(__file__), "runtime_videos", "baked_baguette.avi"
            )
            self.video_on = "baked_baguette"
        elif src == "raw_dough":
            self.video_path = os.path.join(
                os.path.dirname(__file__), "runtime_videos", "raw_dough.avi"
            )
            self.video_on = "raw_dough"
        elif src == "donut":
            self.video_path = os.path.join(
                os.path.dirname(__file__), "runtime_videos", "donut.mp4"
            )
            self.video_on = "donut"

        self.cti_file_location = None

    def set_model_selection(self, model_name):
        valid_models = ["raw_dough", "baked_baguette", "generic", "donut"]
        if model_name in valid_models:
            self.model_selection = model_name
        else:
            raise ValueError("Invalid model selection")

    def get_config(self):
        if self.model_selection:
            engine = self.get_modlel_path()
        return {
            "input_type": self.input_type,
            "model_selection": self.model_selection,
            "video_path": self.video_path,
            "cti_file_location": self.cti_file_location,
            "VIDEO_ENGINE_PATH": engine,
            "VIDEO_INPUT_PATH": self.video_path,
            "VIDEO_MODE": self.video_on,
        }

    def get_cti(self):
        return self.cti_file_location

    def get_modlel_path(self):
        if self.model_selection == "raw_dough":
            return os.path.join(
                os.path.dirname(__file__), "runtime_models", "raw_dough.engine"
            )
        elif self.model_selection == "baked_baguette":
            return os.path.join(
                os.path.dirname(__file__), "runtime_models", "baked_baguette.engine"
            )
        elif self.model_selection == "generic":
            return os.path.join(
                os.path.dirname(__file__), "runtime_models", "generic.engine"
            )
        elif self.model_selection == "donut":
            return os.path.join(
                os.path.dirname(__file__), "runtime_models", "donut.engine"
            )


    def validate_config(self):
        if not self.input_type:
            return False, "Input type not configured"
        if not self.model_selection:
            return False, "Model not selected"
        if self.input_type == "camera" and not self.cti_file_location:
            return False, "CTI file location not provided"
        if self.input_type == "video" and not self.video_path:
            return False, "Video path not provided"
        return True, "Configuration valid"
