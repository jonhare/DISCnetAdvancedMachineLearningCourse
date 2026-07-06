import marimo

__generated_with = "0.23.11"
app = marimo.App(
    width="full",
    layout_file="layouts/segmentation_demo.slides.json",
)


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import numpy as np

    from torchvision.io.image import decode_image
    from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
    from torchvision.transforms.functional import to_pil_image

    from wigglystuff import WebcamCapture

    return FCN_ResNet50_Weights, WebcamCapture, fcn_resnet50, mo, to_pil_image


@app.cell
def _(WebcamCapture, mo):
    webcam = mo.ui.anywidget(WebcamCapture())
    return (webcam,)


@app.cell
def _(FCN_ResNet50_Weights, fcn_resnet50):
    model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
    model.eval()
    model = model.to('mps')
    tf = FCN_ResNet50_Weights.DEFAULT.transforms()
    class_to_idx = {cls: idx for (idx, cls) in enumerate(FCN_ResNet50_Weights.DEFAULT.meta["categories"])}
    return class_to_idx, model, tf


@app.cell
def _(class_to_idx, mo, model, tf, to_pil_image, webcam):
    contents=[mo.md("# `fcn_resnet50` demo"), webcam]

    if webcam.image_base64:
        pil_image = webcam.get_pil().convert('RGB')
        img = tf(pil_image).to('mps')
        prediction = model(img.unsqueeze(0))["out"].squeeze(0)

        normalized_masks = prediction.softmax(dim=0)
        mask = normalized_masks[class_to_idx["person"]]
        contents.append(to_pil_image(mask))

    webcam.capturing = True
    mo.vstack(contents, align="center")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
