import numpy as np
import torch
import sys
import time
import random


TOTAL_BAR_LENGTH = 65
last_time = time.time()

def progress_bar(current, total, msg=None):
    global last_time
    if total <= 0:
        if msg:
            sys.stdout.write(f"\r{msg}\n")
            sys.stdout.flush()
        return
    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    bar = '=' * cur_len + '.'*(TOTAL_BAR_LENGTH - cur_len - 1)
    sys.stdout.write(f'\r[{bar}]')
    if msg:
        sys.stdout.write(f'\r{msg}')
    if current == total - 1:
        sys.stdout.write('\n')
    sys.stdout.flush()


