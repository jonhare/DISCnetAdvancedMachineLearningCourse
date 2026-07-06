import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full", layout_file="layouts/modern-archs.slides.json")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn

    from torchwatcher.drawing import draw_graph_pretty, draw_graph
    from torchvision.models import convnext_tiny, vit_b_16, swin_t
    from torchvision.models.convnext import Conv2dNormActivation
    from torchwatcher.interjection import trace

    import numpy as np

    def print(s):
        mo.output.append(mo.Html(f'<pre>{s}</pre>'))

    return (
        convnext_tiny,
        draw_graph_pretty,
        mo,
        nn,
        print,
        swin_t,
        torch,
        trace,
        vit_b_16,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Modern Vision Architectures
    ## And a few other things
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## NN Convolution Recap

    - A 2D convolution uses $C_{out}$ 3D kernels which are applied across all channels
        - each kernel is $(C_{in}, K_H, K_W)$
        - they are represented as a single 4D tensor $(C_{out}, C_{in}, K_H, K_W)$
    """)
    return


@app.cell
def _(nn, print, torch):
    _x = torch.rand(1, 3, 32, 32)
    _kernels = torch.rand(4, 3, 3, 3)

    _y = nn.functional.conv2d(_x, _kernels, stride=1, padding=1, dilation=1)
    print("What shape is `_y`?")
    # print(_y.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1x1 Convolution Implementation
    """)
    return


@app.cell
def _(nn, print, torch):
    B = 1
    C_in = 3
    C_out = 16
    size = 32

    x = torch.rand(B, C_in, size, size)
    kernel = torch.rand(C_out, C_in, 1, 1)

    # Module version
    conv = nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0, bias=False)
    conv.weight.data = kernel
    y1 = conv(x)

    # Functional version
    y2 = nn.functional.conv2d(x, kernel, stride=1, padding=0, bias=None)
    print("module and functional are the same: " + str(torch.allclose(y1, y2)))

    # Matrix version
    kernelr = kernel.view(16, -1)
    xr = x.view(B, C_in, -1).permute(0, 2, 1) # B, HW, C_in
    y3 = (xr @ kernelr.T).permute(0, 2, 1).view(B, C_out, size, size)
    print("module and matrix are the same: " + str(torch.allclose(y1, y3)))

    # einsum version
    y4 = torch.einsum('bchw, ocij -> bohw', x, kernel)
    print("module and einsum are the same: " + str(torch.allclose(y1, y4)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LayerNorm

    Normalise a tensor $x$ according to

    $$y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta$$

    $\gamma$ and $\beta$ are learned (they are optional and can be omitted completely with the `elementwise_affine` flag).

    $\mathrm{E}[x]$ and $\mathrm{Var}[x]$ are computed over the last D dimensions of x. Note that the computation is broadcast (applied independently) over the remaining dimensions.
    """)
    return


@app.cell
def _(nn, print, torch):
    # Sequence Example
    batch, sentence_length, embedding_dim = 20, 5, 10
    embedding = torch.randn(batch, sentence_length, embedding_dim)

    # LayerNorm is just over the embedding dimension (dim -1)
    # - each embedding is normalised
    # - the affine parameters transform _all_ embeddings with the same transform
    layer_norm = nn.LayerNorm(embedding_dim)

    print(layer_norm.weight.shape)
    print(layer_norm.bias.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### LayerNorm2d

    - Used in modern CNNs instead of BatchNorm
    - A batch of images is typically (batch, channels, height, width)
    - LayerNorm2d is just over the channel dimension
      - Achieved by permuting the channels to be last (batch, height, width, channels), applying standard LayerNorm to the last dimension, and permuting back
    """)
    return


@app.cell
def _(mo, torch):
    import matplotlib.pyplot as plt

    mo.output.append(mo.md("""## GELU Activations
    <center>
    """))

    _xs = torch.linspace(-5, 5, 100)
    _relus = torch.relu(_xs)
    _gelus = torch.nn.functional.gelu(_xs)

    fig = plt.figure(figsize=(5,5))
    plt.plot(_xs, _relus)
    plt.plot(_xs, _gelus)
    plt.ylim(-5, 5)
    plt.gca().spines['left'].set_position(('data', 0.0))
    plt.gca().spines['bottom'].set_position(('data', 0.0))
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.legend(['ReLU', 'GELU'])
    mo.output.append(fig)
    mo.output.append(mo.md("""</center>"""))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Depthwise separable convolution

    - Standard 2D convolution:
      - Parameters: $C_{out} \times C_{in} \times K_H \times K_W$ (+ $C_{out}$ if there is bias)
      - FLOPs: $\sim \mathcal{O}(H_{out} \times W_{out} \times K_H \times K_W \times C_{in} \times C_{out})$
    - What if instead we apply one 2D kernel per channel?
      - $C_{out} == C_{in}$
      - Parameters: $C_{in} \times K_H \times K_W$
      - FLOPs: $\sim \mathcal{O}(H_{out} \times W_{out} \times K_H \times K_W \times C_{in})$
      - But, we've lost capacity by not allowing channels to change or mix, so follow up with a $1 \times 1$ convolution:
        - Parameters: $C_{out} \times C_{in} \times 1 \times 1$ (+ $C_{out}$ if there is bias)
        - FLOPs: $\sim \mathcal{O}(H_{out} \times W_{out} \times 1 \times 1 \times C_{in} \times C_{out})$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Depthwise separable convolution

    - The depthwise spatial step costs only $\frac{1}{K_H K_W}$ as much as a standard convolution
    - The full depthwise-separable block also includes the $1 \times 1$ channel-mixing step
      - FLOP ratio vs standard convolution is roughly $\frac{1}{C_{out}} + \frac{1}{K_H K_W}$
      - For $C_{out}=32$ and $7\times7$ kernels: $\sim 5\%$ of the standard convolution cost
    - But, at the expense of _capacity_...
    """)
    return


@app.cell
def _(nn, print, torch):
    Cin = 3
    Cout = 32
    H, W = 64, 64
    KH, KW = 7, 7

    K_spatial = torch.rand(Cin, 1, KH, KW)
    K_channel = torch.rand(Cout, Cin, 1, 1)

    _x = torch.rand(1, Cin, H, W)

    # Groups=Cin tells the implementation that there is one 2d kernel per channel
    _y_inter = nn.functional.conv2d(_x, K_spatial, groups=Cin, padding=3) # spatial
    _y = nn.functional.conv2d(_y_inter, K_channel) # channel mixing

    print(_y_inter.shape)
    print(_y.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Stochastic Depth

    - Recall that dropout removes (zeros out) neurons randomly during training (not during inference!)
      - acts as a regulariser to stop overfitting
    - Stochastic depth does the same, but it randomly zeros out entire tensors rather than just individual elements
      - Obviously this doesn't make sense in a standard feed-forward network, but with residual connections it makes sense:
      - Consider $y = f(x) + x$ and add the stochastic depth to the residual $f(x)$: $y = \mathrm{stochastic\_depth}(f(x)) + x$
        - During training of a network built of a sequential stack of residual blocks this has the effect of dynamically reducing the network depth
    """)
    return


@app.cell
def _(nn, print, torch):
    from torchvision.ops import stochastic_depth

    class SDDemo(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def forward(self, x):
            residual = stochastic_depth(
                self.weight * x,
                p=0.5,
                mode='batch',
                training=self.training,
            )
            return residual + x

    _net = nn.Sequential(SDDemo(), SDDemo(), SDDemo())
    _net.train()
    for i in range(5):
        print(_net(2).item())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The VisionTransformer

    - Breakthrough in using transformer-based models on image data
    - Super performance, but very data hungry due to reduced inductive biases
    - Turns an image into a sequence of patch-tokens using a **strided convolution**, then follows a standard transformer-based classification pipeline
    """)
    return


@app.cell
def _(draw_graph_pretty, mo, torch, trace, vit_b_16):
    from torchvision.models.vision_transformer import EncoderBlock
    image_size=224
    mo.vstack([
        mo.md("# VisionTransformer: `vit_b_16`"),
        mo.Html('<center>'+ draw_graph_pretty(trace(vit_b_16(image_size=image_size), tracer_kwargs={'leaf_modules': [EncoderBlock]}), torch.empty(1,3,image_size,image_size), show_extra_info=True).create_svg().decode('utf-8') + '</center>')
    ])
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("# What is the `VisionTransformer` doing?"),
        mo.ui.code_editor("""def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)""")
    ])
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("# Position Embeddings"),
        mo.ui.code_editor("""class Encoder(nn.Module):
        def __init__(...):
            ...
            self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
            ...

        def forward(self, input: torch.Tensor):
            input = input + self.pos_embedding
            return self.ln(self.layers(self.dropout(input)))""")
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ViT Position Embeddings

    - Fully learned
    - **Added** to the respective patch embedding
    - **Fixed** at one per patch plus the `CLS` token (e.g. number of position embeddings == seq_length)
        + means that we cannot change the input size
            + (can potentially interpolate as a workaround)
    """)
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("# What is the `VisionTransformer` doing?"),
        mo.ui.code_editor("""def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x""")
    ])
    return


@app.cell
def _(draw_graph_pretty, mo, torch, trace, vit_b_16):
    mo.vstack([
        mo.md("# VisionTransformer: `vit_b_16 EncoderBlock` layers"),
        mo.Html('<center>' + draw_graph_pretty(trace(vit_b_16().encoder.layers[0]), torch.empty(1,197,768), show_extra_info=True).create_svg().decode('utf-8') + '</center>')
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Swin Transformer

    - ViT uses global attention over all patch tokens
      - very expressive
      - attention cost scales quadratically with sequence length
      - intermediate features stay relatively flat compared with CNN feature pyramids
    - Swin makes the transformer feel more CNN-like:
      - local self-attention inside small windows
      - shifted windows so neighbouring windows can exchange information
      - hierarchical stages that downsample spatial resolution and increase channels
    - This gives a practical backbone for classification, detection, and segmentation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md("# Global vs Windowed Attention"),
        mo.image(
            src="images/swin/swin-t-global-vs-window-vs-shifted-window-self-attention.jpg",
            alt="Comparison of global, window, and shifted-window self-attention",
            width="92vw",
            style={"max-height": "68vh", "object-fit": "contain"},
            caption="Source: ml-digest.com",
        ),
    ], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md("# Shifted Windows"),
        mo.image(
            src="images/swin/shifted_window.png",
            alt="Swin Transformer shifted-window attention pattern",
            width="92vw",
            style={"max-height": "68vh", "object-fit": "contain"},
            caption="Source: Swin Transformer paper",
        ),
    ], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md("# Hierarchical Feature Maps"),
        mo.image(
            src="images/swin/hierarchy.png",
            alt="Swin Transformer hierarchical feature map architecture",
            width="92vw",
            style={"max-height": "68vh", "object-fit": "contain"},
            caption="Source: Swin Transformer paper",
        ),
    ], align="center")
    return


@app.cell
def _(draw_graph_pretty, mo, swin_t, torch, trace):
    from torchvision.models.swin_transformer import SwinTransformerBlock

    mo.vstack([
        mo.md("# Swin Transformer: `swin_t`"),
        mo.Html('<center>' + draw_graph_pretty(trace(swin_t(), tracer_kwargs={'leaf_modules': [SwinTransformerBlock]}), torch.empty(1,3,224,224), show_extra_info=True).create_svg().decode('utf-8') + '</center>')
    ])
    return


@app.cell
def _(draw_graph_pretty, mo, swin_t, torch, trace):
    from torchvision.models.swin_transformer import MLP, ShiftedWindowAttention
    mo.vstack([
        mo.md("# Swin Transformer: `swin_t SwinTransformerBlock`"),
        mo.Html('<center>' + draw_graph_pretty(trace(swin_t().features[1][1], tracer_kwargs={'leaf_modules': [ShiftedWindowAttention, MLP]}), torch.empty(1,56,56,96), show_extra_info=True).create_svg().decode('utf-8') + '</center>')
    ])
    return


@app.cell
def _(draw_graph_pretty, mo, swin_t, torch, trace):
    mo.vstack([
        mo.md("# Swin Transformer: `swin_t ShiftedWindowAttention layers`"),
        mo.Html('<center>'+draw_graph_pretty(trace(swin_t().features[1][1].attn), torch.empty(1,56,56,96), show_extra_info=True).create_svg().decode('utf-8') + '</center>')
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Modern CNNs
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CNNs are not dead yet!

    - Transformers' lack of architectural inductive bias means they require huge amounts of data
      - A lot of compute is spent learning things we know are desirable (e.g. translation equivariance)
      - But transformers do have an advantage in terms of global attention
    - Traditional CNNs migrated to small kernels
      - Depth used to grow receptive fields and get global integration (but at the cost of resolution)
      - Modern CNNs moved to larger (DW-separable) kernels to increase RF
        - Expansion-compression trick applied to mitigate loss in capacity
        - Plus some other ideas from transformers
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ConvNeXt

    - ConvNeXt asks: what if we modernise a ResNet using transformer-era design choices?
    - Key changes:
      - larger depthwise kernels for spatial mixing
      - inverted bottlenecks: expand channels, mix, then compress
      - LayerNorm instead of BatchNorm in the main blocks
      - GELU activations and fewer activation/norm layers overall
      - stochastic depth for training deep residual stacks
    - The result is still a convolutional network, but with many of the practical lessons learned from ViTs.
    """)
    return


@app.cell
def _(convnext_tiny, draw_graph_pretty, mo, torch, trace):
    mo.vstack([
        mo.md("# ConvNeXt: `convnext_tiny`"),
        mo.Html('<center>' + draw_graph_pretty(trace(convnext_tiny()), torch.empty(1,3,224,224), show_extra_info=True).create_svg().decode('utf-8') + '</center>')
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # RepLKNet

    - ConvNeXt pushed us away from the $3 \times 3$ kernels that were popularised by the classic CNN architectures
    - RepLKNet pushed the idea of larger kernels much further
      - e.g. $31 \times 31$ depthwise kernels in the main spatial mixing path
    - Motivation:
      - CNNs are naturally local
      - Transformers get long-range interaction via attention
      - Very large convolution kernels can give CNNs a much larger effective receptive field
    - The key claim: large-kernel CNNs can recover some global-context behaviour without abandoning convolutional structure.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # RepLKNet: making huge kernels trainable

    - Large kernels are expensive and can be awkward to optimise directly
    - RepLKNet uses **structural re-parameterisation**:
      - during training, combine a large depthwise kernel with smaller auxiliary branches
      - after training, fold the branches into a single large convolution for inference
    - This keeps the inference graph simple while making optimisation easier
    - Takeaway for the architecture story:
      - modern CNNs did not just borrow normalisation and training tricks from transformers
      - they also re-opened old questions about receptive field size and spatial mixing
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Other important directions

    - **UNet-style encoder-decoder models**
      - skip connections preserve spatial detail for segmentation and generation
      - the pattern remains central in medical imaging, diffusion models, and dense prediction
    - **Contrastive image-text pretraining**
      - CLIP-style models changed what a "vision backbone" can be used for
      - zero-shot and retrieval behaviour come from the training objective, not only the architecture
    - **Masked image modelling and foundation segmentation**
      - MAE-style pretraining made self-supervised ViTs much more practical
      - SAM-style promptable segmentation showed the value of very broad pretraining for dense vision tasks
    """)
    return


if __name__ == "__main__":
    app.run()
