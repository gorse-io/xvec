import ctypes, json, pathlib
root=pathlib.Path(__file__).resolve().parent
lib=ctypes.CDLL(str(root/'native/linux_amd64/libzvec_c_api.so'))
def fn(name, result, args):
 f=getattr(lib,name);f.restype=result;f.argtypes=args;return f
ptr=ctypes.c_void_p
version=fn('zvec_get_version',ctypes.c_char_p,[])().decode()
config=fn('zvec_config_data_create',ptr,[])()
ratio=fn('zvec_config_data_get_brute_force_by_keys_ratio',ctypes.c_float,[ptr])(config)
fn('zvec_config_data_destroy',None,[ptr])(config)
params=fn('zvec_index_params_create',ptr,[ctypes.c_uint32])(1)
assert params
assert fn('zvec_index_params_set_quantize_type',ctypes.c_int32,[ptr,ctypes.c_uint32])(params,1)==0
quantize=fn('zvec_index_params_get_quantize_type',ctypes.c_uint32,[ptr])(params)
rotate=fn('zvec_index_params_get_quantizer_enable_rotate',ctypes.c_bool,[ptr])(params)
fn('zvec_index_params_destroy',None,[ptr])(params)
assert quantize==1 and rotate==False
print(json.dumps({'native_version':version,'fp16_quantization_enum':quantize,'fp16_rotate':rotate,'brute_force_by_keys_ratio':ratio},indent=2))
