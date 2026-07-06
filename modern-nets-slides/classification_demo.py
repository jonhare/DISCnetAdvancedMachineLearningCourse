import marimo

__generated_with = "0.23.11"
app = marimo.App(layout_file="layouts/classification_demo.slides.json")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import numpy as np

    from torchvision.models import convnext_tiny, vit_b_16, swin_t
    from torchvision.models.convnext import Conv2dNormActivation
    from torchvision.models import ViT_B_16_Weights

    from wigglystuff import WebcamCapture

    return ViT_B_16_Weights, WebcamCapture, mo, vit_b_16


@app.cell
def _(WebcamCapture, mo):
    webcam = mo.ui.anywidget(WebcamCapture())
    return (webcam,)


@app.cell
def _(ViT_B_16_Weights, vit_b_16):
    vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    tf = ViT_B_16_Weights.DEFAULT.transforms()
    vit = vit.to("mps")
    return tf, vit


@app.cell
def _(ViT_B_16_Weights, mo, tf, vit, webcam):
    if webcam.image_base64:
        pil_image = webcam.get_pil().convert('RGB')
        img = tf(pil_image).to('mps')
        clz = vit(img.unsqueeze(0)).squeeze(0).argmax().cpu().item()
        clz = ViT_B_16_Weights.DEFAULT.meta['categories'][clz]
    else:
        clz = ""

    webcam.capturing = True
    mo.vstack([mo.md("# `vit_b_16` demo"), webcam, mo.md(clz)], align="center")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
