"""Vendored copy of the Kronos model code (MIT License, see LICENSE).

Source: https://github.com/shiyu-coder/Kronos (model/kronos.py, model/module.py).
Only the import paths were adapted so it can live inside this package.
"""
from .kronos import Kronos, KronosPredictor, KronosTokenizer, auto_regressive_inference, calc_time_stamps

__all__ = ["Kronos", "KronosTokenizer", "KronosPredictor", "auto_regressive_inference", "calc_time_stamps"]
