import marimo

__generated_with = "0.23.11"
app = marimo.App(
    width="full",
    layout_file="layouts/keypoints_demo.slides.json",
)


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import numpy as np

    from torchvision.models.detection import (
        KeypointRCNN_ResNet50_FPN_Weights,
        keypointrcnn_resnet50_fpn,
    )
    from torchvision.transforms.functional import pil_to_tensor, to_pil_image
    from torchvision.utils import draw_bounding_boxes, draw_keypoints

    from wigglystuff import WebcamCapture

    return (
        KeypointRCNN_ResNet50_FPN_Weights,
        WebcamCapture,
        draw_bounding_boxes,
        draw_keypoints,
        keypointrcnn_resnet50_fpn,
        mo,
        pil_to_tensor,
        to_pil_image,
        torch,
    )


@app.cell
def _(WebcamCapture, mo):
    webcam = mo.ui.anywidget(WebcamCapture())
    return (webcam,)


@app.cell
def _(KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn):
    model = keypointrcnn_resnet50_fpn(
        weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT,
        box_score_thresh=0.9,
    )
    model.eval()
    model = model.to("mps")
    tf = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT.transforms()
    return model, tf


@app.cell
def _():
    skeleton = [
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 6),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
    ]
    return (skeleton,)


@app.cell
def _(
    draw_bounding_boxes,
    draw_keypoints,
    mo,
    model,
    pil_to_tensor,
    skeleton,
    tf,
    to_pil_image,
    torch,
    webcam,
):
    contents = [webcam]

    if webcam.image_base64:
        pil_image = webcam.get_pil().convert("RGB")

        img = tf(pil_image).to("mps")
        with torch.no_grad():
            prediction = model(img.unsqueeze(0))[0]

        display_img = pil_to_tensor(pil_image)
        keypoints = prediction["keypoints"].cpu()
        boxes = prediction["boxes"].cpu()
        visibility = keypoints[:, :, 2] > 0

        if len(keypoints) > 0:
            pose = draw_keypoints(
                display_img,
                keypoints[:, :, :2],
                connectivity=skeleton,
                colors="cyan",
                radius=4,
                width=3,
                visibility=visibility,
            )
            pose = draw_bounding_boxes(
                pose,
                boxes=boxes,
                colors="red",
                width=4,
            )
            contents.append(to_pil_image(pose))

    webcam.capturing = True
    mo.vstack(
        [
            mo.md("# `keypointrcnn_resnet50_fpn` keypoints demo"),
            mo.hstack(contents, align="center"),
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
