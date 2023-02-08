# Instructions for Installing and Running DeepExplain

Then replace, `src/deepexplain/deepexplain/tensorflow/methods.py` with `MtbQuantCNN/analysis/deepexplain_methods.py`. The latter file contains some changes for compatibility with models trained in Tensorflow v2 and some changes for how the multi-input models are structured. `src/deepexplain/deepexplain/tensorflow/methods.py` will be located in whichever directory the `pip install` command above was run from. 
