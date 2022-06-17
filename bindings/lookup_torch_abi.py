#!/usr/bin/env python3

"""Looks up which ABI the default PyTorch installation uses."""

import torch
print(int(torch._C._GLIBCXX_USE_CXX11_ABI))
