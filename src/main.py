from packet import input, weight

import numpy as np
from time import sleep

def main_target():
    sleep(10)
    input.set((np.random.rand(32, 32) * 255).astype(np.uint8))
    weight.set((np.random.rand(32, 32) * 255).astype(np.uint8))
    sleep(10)
    input.set((np.random.rand(32, 32) * 255).astype(np.uint8))
    sleep(10)
    weight.set((np.random.rand(32, 32) * 255).astype(np.uint8))
    ...
