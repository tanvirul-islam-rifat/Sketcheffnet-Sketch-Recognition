# models/

Trained weights are not committed (`*.pth` is git-ignored). To use the inference CLI:

1. Download `sketchnet_best.pth` from the **Output** tab of the Kaggle notebook that ran the
   pipeline (`/kaggle/working/models/sketchnet_best.pth`, about 12 MB because it includes the optimizer state),
   or train it yourself with `python -m experiments.day5_sketcheffnet`.
2. Put it here as `models/sketchnet_best.pth`.
3. `python -m sketcheffnet.predict my_drawing.png`

To publish the weights, attach the file to a GitHub Release rather than committing it.
