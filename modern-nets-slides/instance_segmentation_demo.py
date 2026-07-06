import marimo

__generated_with = "0.23.11"
app = marimo.App(
    width="full",
    layout_file="layouts/instance_segmentation_demo.slides.json",
)


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import numpy as np

    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_V2_Weights,
        maskrcnn_resnet50_fpn_v2,
    )
    from torchvision.transforms.functional import pil_to_tensor, to_pil_image
    from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

    from wigglystuff import WebcamCapture

    return (
        MaskRCNN_ResNet50_FPN_V2_Weights,
        WebcamCapture,
        draw_bounding_boxes,
        draw_segmentation_masks,
        maskrcnn_resnet50_fpn_v2,
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
def _(MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2):
    model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        box_score_thresh=0.85,
    )
    model.eval()
    model = model.to("mps")
    tf = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT.transforms()
    return model, tf


@app.cell
def _(
    MaskRCNN_ResNet50_FPN_V2_Weights,
    draw_bounding_boxes,
    draw_segmentation_masks,
    mo,
    model,
    pil_to_tensor,
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
        masks = prediction["masks"].cpu().squeeze(1) > 0.5
        boxes = prediction["boxes"].cpu()
        labels = [
            MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT.meta["categories"][int(i)]
            for i in prediction["labels"].cpu()
        ]

        if len(masks) > 0:
            segmented = draw_segmentation_masks(
                display_img,
                masks=masks,
                alpha=0.65,
            )
            segmented = draw_bounding_boxes(
                segmented,
                boxes=boxes,
                labels=labels,
                colors="red",
                width=4,
                font="Helvetica",
                font_size=30,
            )
            contents.append(to_pil_image(segmented))

    webcam.capturing = True
    mo.vstack(
        [
            mo.md("# `maskrcnn_resnet50_fpn_v2` instance segmentation demo"),
            mo.hstack(contents, align="center"),
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
