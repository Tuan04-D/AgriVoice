"""BioCLIP 2 backbone, LoRA adapters and the linear classification head."""

import math

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import utils


def load_backbone(model_cfg, device):
    """Load the open_clip visual tower and its normalization statistics."""
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_cfg["name"], pretrained=model_cfg["pretrained"]
    )
    normalize = next(
        transform for transform in preprocess.transforms
        if isinstance(transform, torchvision.transforms.Normalize)
    )
    visual = clip_model.visual
    del clip_model
    for parameter in visual.parameters():
        parameter.requires_grad = False
    mean = [float(value) for value in normalize.mean]
    std = [float(value) for value in normalize.std]
    utils.get_logger().info(
        "Backbone %s ready: embed_dim=%d, blocks=%d", model_cfg["name"], visual.output_dim,
        len(visual.transformer.resblocks),
    )
    return visual.to(device), mean, std


def build_head(embed_dim, num_classes, device):
    """Create the linear classification head."""
    return nn.Linear(embed_dim, num_classes).to(device)


def head_logits(head, features, device):
    """Return logits for pre-computed embeddings."""
    head.eval()
    with torch.no_grad():
        return torch.cat([head(chunk.to(device)).float().cpu() for chunk in features.split(4096)])


def load_head_state(path, embed_dim, device):
    """Load a classification head from a training checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = {
        key.split("head.", 1)[1]: value
        for key, value in payload["state_dict"].items() if key.startswith("head.")
    }
    head = build_head(embed_dim, state["bias"].numel(), device)
    head.load_state_dict(state)
    return head, payload


class LoRALinear(nn.Module):
    """Linear layer with a frozen base weight and a trainable low-rank update."""

    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, inputs):
        """Apply the frozen base layer plus the low-rank update."""
        update = self.dropout(inputs) @ self.lora_a.t() @ self.lora_b.t()
        return self.base(inputs) + update * self.scale


def inject_lora(visual, lora_cfg):
    """Replace the MLP projections of the last blocks with LoRA layers."""
    blocks = visual.transformer.resblocks
    first = max(0, len(blocks) - lora_cfg["last_n_blocks"])
    names = []
    for block_index in range(first, len(blocks)):
        mlp = blocks[block_index].mlp
        for target in lora_cfg["targets"]:
            base = getattr(mlp, target)
            if not isinstance(base, nn.Linear):
                raise TypeError(
                    f"resblocks.{block_index}.mlp.{target} is {type(base).__name__}, "
                    "expected nn.Linear"
                )
            lora = LoRALinear(base, lora_cfg["rank"], lora_cfg["alpha"], lora_cfg["dropout"])
            setattr(mlp, target, lora.to(base.weight.device))
            names.append(f"transformer.resblocks.{block_index}.mlp.{target}")
    for name, parameter in visual.named_parameters():
        parameter.requires_grad = "lora_" in name
    return names


class DiagnosisModel(nn.Module):
    """Visual backbone with L2-normalized embeddings and a linear head."""

    def __init__(self, visual, head):
        super().__init__()
        self.visual = visual
        self.head = head

    def forward(self, images):
        """Return class logits for a batch of images."""
        return self.head(F.normalize(self.visual(images), dim=-1))


def trainable_state(module):
    """Return the state dict entries belonging to trainable parameters."""
    names = {name for name, parameter in module.named_parameters() if parameter.requires_grad}
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items() if name in names
    }


def probe_batch_size(diagnosis_model, device, use_amp, lora_cfg, image_size, num_classes):
    """Return the largest candidate batch size that fits in GPU memory."""
    candidates = lora_cfg["batch_size_candidates"]
    if device.type != "cuda":
        return candidates[-1]
    logger = utils.get_logger()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    diagnosis_model.train()
    for batch_size in candidates:
        images = labels = logits = loss = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            images = torch.randn(batch_size, 3, image_size, image_size, device=device)
            labels = torch.randint(0, num_classes, (batch_size,), device=device)
            for _ in range(2):
                with utils.autocast_context(device, use_amp):
                    logits = diagnosis_model(images)
                loss = F.cross_entropy(logits.float(), labels)
                loss.backward()
                diagnosis_model.zero_grad(set_to_none=True)
            fraction = torch.cuda.max_memory_allocated() / total_memory
            logger.info(
                "[lora] batch probe %d: peak %.1f%% of GPU memory", batch_size, 100 * fraction
            )
            if fraction <= lora_cfg["max_memory_fraction"]:
                return batch_size
        except torch.cuda.OutOfMemoryError:
            logger.info("[lora] batch probe %d: out of memory", batch_size)
        finally:
            del images, labels, logits, loss
            diagnosis_model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    raise RuntimeError("No candidate batch size fits in GPU memory")


def cosine_with_warmup(optimizer, warmup_steps, total_steps, min_ratio):
    """Return a linear warmup followed by cosine decay scheduler."""
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
