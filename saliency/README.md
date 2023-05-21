# Instructions for Installing and Running DeepExplain

Then replace, `src/deepexplain/deepexplain/tensorflow/methods.py` with `MtbQuantCNN/saliency/deepexplain_methods.py`. The latter file contains some changes for compatibility with models trained in Tensorflow v2 and some changes for how I structured multi-input models. `src/deepexplain/deepexplain/tensorflow/methods.py` will be located in whichever directory the `pip install` command above was run from. 
