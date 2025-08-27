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

    def set_camera_config(self, cti_file_location):
        self.input_type = "camera"
        self.cti_file_location = cti_file_location
        self.video_path = None

    def set_video_config(self, video_path):
        self.input_type = "video"
        self.video_path = video_path
        self.cti_file_location = None

    def set_model_selection(self, model_name):
        valid_models = ["raw_dough", "baked_baguette", "generic"]
        if model_name in valid_models:
            self.model_selection = model_name
        else:
            raise ValueError("Invalid model selection")

    def get_config(self):
        return {
            "input_type": self.input_type,
            "model_selection": self.model_selection,
            "video_path": self.video_path,
            "cti_file_location": self.cti_file_location
        }

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