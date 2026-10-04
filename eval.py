from ultralytics import YOLOE
path = "checkpoints/YOLOE11_EvaGen.pt"
model = YOLOE(path)
model.fuse()

# from DroneAmbulanceAI.vista.models.sam.src.sam3_model import Sam3Model
# model = Sam3Model()

# from vista.moondream import MoonDream
# from ultralytics import YOLO

# from vista.omdetturbo import OmDetTurbo

# model = MoonDream()
# model.set_classes(["crashed_car", "person", "car"])

# model = OmDetTurbo()
# model.set_classes(["crashed_car", "person", "car"])


data_path = "data/VistaSynth/data.yaml"
results = model.val(data=data_path, split="test")
