"""Default locations, relative to the repository root.

Environment variables override them:
    GDECIL_DATASETS_DIR   raw image datasets    (default: <repo>/datasets)
    GDECIL_FEATURES_DIR   extracted features    (default: <repo>/features)
    GDECIL_RESULTS_DIR    accuracy matrices     (default: <repo>/results)
"""
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DATASETS_DIR = os.environ.get('GDECIL_DATASETS_DIR', os.path.join(REPO_ROOT, 'datasets'))
FEATURES_DIR = os.environ.get('GDECIL_FEATURES_DIR', os.path.join(REPO_ROOT, 'features'))
RESULTS_DIR = os.environ.get('GDECIL_RESULTS_DIR', os.path.join(REPO_ROOT, 'results'))
