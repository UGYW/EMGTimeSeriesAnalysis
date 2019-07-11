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
        pass
    return res