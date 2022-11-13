def replace_nan(input):
    return [e if e == e else 0 for e in input]
  

def unpack_array(input):
    return [e[0] if hasattr(e, '__len__') else e for e in input]