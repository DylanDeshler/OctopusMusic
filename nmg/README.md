Vendored model code from the Neural Music Generator Space
(huggingface.co/spaces/ddeshler/Neural_Music_Generator, commit in SOURCE_COMMIT),
copied unmodified so the GPU server needs only this repo plus checkpoints.
If you change the model code there, re-copy these files (or set NMG_PATH to a
checkout to use it directly). Checkpoints (*.pt) are NOT here: put them in this
folder or let the server download them from MODEL_REPO.
