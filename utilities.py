import os

def str_to_seconds(timestamp_string):
    if type(timestamp_string) != str:
        timestamp_string = timestamp_string.values[0]  # in case of using numpy apply
    res = -1
    if len(timestamp_string) <= 2:  # case '-1'
        pass
    elif len(timestamp_string) == 5:  # case 'MM:SS'
        m, s = timestamp_string.split(":")
        res = int(m) * 60 + int(s)
    elif len(timestamp_string) == 7:  # case 'MM:SS:MSMS'
        m, s, _ = timestamp_string.split(":")
        res = int(m) * 60 + int(s)
    else:
        res = float(timestamp_string)
    return res

def get_file_paths(path_to_main_folder):
    contents = os.listdir(path_to_main_folder)
    file_paths = [path_to_main_folder + "/" + i for i in contents if ".txt" in i]
    return file_paths