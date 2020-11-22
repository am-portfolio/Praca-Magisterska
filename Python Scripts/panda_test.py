# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics
from timeit import default_timer as timer
from pymodules.utilities import *
from pymodules.mlrnhelpers import *
from pymodules.mlevaluator import *



x = summaryToTrainableParams('test2.txt')