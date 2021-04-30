#!/usr/bin/env python

"""Looks up which ABI the default PyTorch installation uses."""

try:
    import torch
except ImportError:
    print(1)
else:
    print(int(torch._C._GLIBCXX_USE_CXX11_ABI))
