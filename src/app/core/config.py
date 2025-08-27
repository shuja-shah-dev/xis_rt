CONFIG = {
    "cameras": {
        "video": {
            "name": "Video As Webcam",
            "requires_cti": False,
            "options": ["Raw Dough", "Baked Baguette"]  
        },
        "allied": {
            "name": "Allied Camera",
            "requires_cti": True,
            "cti_path": "./cti_files/VimbaUSBTL.cti"
        },
        "ids": {
            "name": "IDS Camera",
            "requires_cti": True,
            "cti_path": "./cti_files/ids_driver.cti"
        }
    },
    "models": {
        "generic": {
            "category": "Generic",
            "items": [
                {"id": "general", "name": "General Model", "path": "./models/general_model.onnx"}
            ]
        },
        "baguette": {
            "category": "Baguette",
            "items": [
                {"id": "digital_twin", "name": "Digital Twin Model", "path": "./models/baguette_digital_twin.onnx"},
                {"id": "real_data", "name": "Real Data Model", "path": "./models/baguette_real_data.onnx"}
            ]
        },
        "raw": {
            "category": "Raw",
            "items": [
                {"id": "digital_twin_raw", "name": "Digital Twin Model", "path": "./models/raw_digital_twin.onnx"}
            ]
        }
    }
}
