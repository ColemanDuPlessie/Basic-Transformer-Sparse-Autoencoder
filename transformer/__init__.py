import platform

import torch as t

device = "cuda" if t.cuda.is_available() else "cpu"

if "ncsa" in platform.node() and device == "cuda":
    debug = False
else:
    debug = True

artifacts_prefix = ""
if platform.node() == 'morin-y':
    artifacts_prefix = "../"
    deepspeed_config_location = "/home/dyusha/research/dictionary_learning/dsconfig.json"
elif t.cuda.is_available():
    deepspeed_config_location = "dsconfig.json"
else:
    deepspeed_config_location = None
