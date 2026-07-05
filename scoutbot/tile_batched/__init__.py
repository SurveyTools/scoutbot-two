def __getattr__(name):
    if name == 'Yolov8DetectionModel':
        from .main import Yolov8DetectionModel

        globals()['Yolov8DetectionModel'] = Yolov8DetectionModel
        return Yolov8DetectionModel
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
