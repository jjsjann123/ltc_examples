#!/bin/bash
LTC_TS_CUDA=1 LTC_IR_DEBUG=1 LTC_SAVE_TENSORS_FILE="file" python ./repro.py &> cache
