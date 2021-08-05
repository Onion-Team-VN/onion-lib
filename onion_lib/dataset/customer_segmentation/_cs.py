import pathlib
import os 

def get_dataset_path(data_name):
    current_directory = pathlib.Path(__file__).parent.resolve()
    return os.path.join(current_directory,data_name) 