import marimo

__generated_with = "0.23.11"
app = marimo.App(
    width="full",
    layout_file="layouts/detection_demo.slides.json",
)


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import numpy as np

    from torchvision.io.image import decode_image
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
    from torchvision.utils import draw_bounding_boxes
    from torchvision.transforms.functional import to_pil_image

    from wigglystuff import WebcamCapture

    return (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        WebcamCapture,
        draw_bounding_boxes,
        fasterrcnn_resnet50_fpn_v2,
        mo,
        to_pil_image,
    )


@app.cell
def _(WebcamCapture, mo):
    webcam = mo.ui.anywidget(WebcamCapture())
    return (webcam,)


@app.cell
def _(FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2):
    model = model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, box_score_thresh=0.9)
    model.eval()
    model = model.to('mps')
    tf = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.transforms()
    return model, tf


@app.cell
def _(
    FasterRCNN_ResNet50_FPN_V2_Weights,
    draw_bounding_boxes,
    mo,
    model,
    tf,
    to_pil_image,
    webcam,
):
    contents=[webcam]

    if webcam.image_base64:
        pil_image = webcam.get_pil().convert('RGB')

        img = tf(pil_image).to('mps')
        prediction = model(img.unsqueeze(0))[0]

        labels = [FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.meta["categories"][i] for i in prediction["labels"]]
        box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                                  labels=labels, colors="red",
                                  width=4, font="Helvetica", font_size=30)
        contents.append(to_pil_image(box))

    webcam.capturing = True
    mo.vstack([mo.md("# `fasterrcnn_resnet50_fpn_v2` detection demo"), mo.hstack(contents, align="center")])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
