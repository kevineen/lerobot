## Paper

https://tonyzhaozh.github.io/aloha

## Vision backbones

By default, ACT encodes camera images with a torchvision ResNet (`resnet18`, ImageNet-pretrained).
The `vision_backbone` option also accepts Hugging Face models (e.g. DINOv2, DINOv3) and timm
models. The number of output feature channels is detected automatically, so switching backbones
requires no other architecture changes.

### Choosing a backbone

| `vision_backbone`         | Source                 | Example                                                            |
| ------------------------- | ---------------------- | ------------------------------------------------------------------ |
| `resnet18`, `resnet50`, … | torchvision (default)  | `resnet18`                                                         |
| HF model id or local path | `transformers`         | `facebook/dinov2-base`, `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| `timm:<model_name>`       | `timm`                 | `timm:vit_base_patch14_dinov2.lvd142m`                             |

Dependencies: Hugging Face backbones need `pip install -e ".[transformers]"` (included in
`lerobot[all]`), timm backbones need `pip install -e ".[timm]"`.

> **Note:** the `facebook/dinov3-*` repos are gated — accept the license on the Hub and run
> `huggingface-cli login` before training. DINOv2 repos are public.

### Related config fields

- `pretrained_backbone_weights` — where the weights come from:
  - torchvision backbones: a torchvision weights enum name (default
    `ResNet18_Weights.IMAGENET1K_V1`); `null` means random init.
  - HF / timm backbones: weights load from `vision_backbone` itself by default (a leftover
    torchvision enum value is ignored). Set it to another model id / pretrained tag to mix and
    match, or `null` / `"random"` for random init.
- `freeze_vision_backbone` — freeze the backbone: its parameters are excluded from the optimizer
  and it always runs in eval mode. Training only the rest of the policy is a cheap and often
  effective recipe with strong pretrained ViTs on small datasets.
- `replace_final_stride_with_dilation` — torchvision ResNet backbones only.
- `optimizer_lr_backbone` — learning rate for the (trainable) backbone parameters.

### Notes

- For ViT backbones, images are replicate-padded to a multiple of the patch size and special
  tokens (cls, registers) are stripped, so any input resolution works — e.g. 480x640 wrist-camera
  images with DINOv2 (patch 14) or DINOv3 (patch 16).
- ViT backbones emit far more encoder tokens than a ResNet feature map (≈1.6k vs 300 at 480x640),
  so the transformer cost and memory grow accordingly; `freeze_vision_backbone=true` also skips
  the backbone backward pass.

### Examples

Fine-tune ACT with a DINOv2 backbone:

```bash
lerobot-train \
  --dataset.repo_id=local/my_dataset \
  --policy.type=act \
  --policy.vision_backbone=facebook/dinov2-base \
  --policy.device=cuda \
  --output_dir=outputs/train/act_dinov2
```

Freeze the backbone (train the transformer + action head only):

```bash
lerobot-train ... \
  --policy.vision_backbone=facebook/dinov2-base \
  --policy.freeze_vision_backbone=true
```

The same options work in a training YAML:

```yaml
policy:
  type: act
  vision_backbone: facebook/dinov2-base
  freeze_vision_backbone: true
```

## Citation

```bibtex
@article{zhao2023learning,
  title={Learning fine-grained bimanual manipulation with low-cost hardware},
  author={Zhao, Tony Z and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
  journal={arXiv preprint arXiv:2304.13705},
  year={2023}
}
```
