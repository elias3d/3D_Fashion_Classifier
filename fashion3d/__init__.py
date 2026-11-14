"""fashion3d: shared code for the Maya 3D fashion classifier.

`classes` and `preprocess` only depend on numpy, so they also run inside
Maya's own Python (mayapy). `model` and `inference` need PyTorch.
"""
