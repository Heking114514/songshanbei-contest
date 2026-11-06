import transformers

print("transformers", transformers.__version__)

import infinilm
from infinilm.lib import _infinilm

print("infinilm import ok", _infinilm.__file__)
