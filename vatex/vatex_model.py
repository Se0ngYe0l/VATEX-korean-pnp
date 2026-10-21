import logging
import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList, Instances

from transformers import AutoProcessor, AutoTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
import open_clip
import numpy as np

import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from .modeling.criterion import VideoSetCriterion
from .modeling.matcher import VideoHungarianMatcher
from .modeling.attention import FFNLayer
from .modeling.transformer_decoder.position_encoding import PositionEmbeddingSine1D

from .utils.memory import retry_if_cuda_oom
from .utils.misc import NestedTensor
from .utils.noun_phrase import extract_noun_phrase#, correct_sentence
from .utils.noun_phrase_ko import extract_noun_phrase_ko

from einops import rearrange, repeat

logger = logging.getLogger(__name__)
import spacy

@META_ARCH_REGISTRY.register()
class VATEX(nn.Module):
    """
    Main class for referring segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # video
        num_frames,
        # multilingual (Korean) CLIP prior/text-tower, see class docstring
        multilingual_clip: bool = False,
        multilingual_clip_model: str = "xlm-roberta-base-ViT-B-32",
        multilingual_clip_pretrained: str = "laion5b_s13b_b90k",
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.num_frames = num_frames

        # Build Text Encoder
        hidden_dim = 256
        self.pe_text = PositionEmbeddingSine1D(hidden_dim, normalize=True)

        self.multilingual_clip = multilingual_clip
        if multilingual_clip:
            # Training-free-style swap: an open_clip checkpoint whose text
            # tower is HF-backed (XLM-RoBERTa), so Korean text can be fed
            # directly with no live translation. Both text_dim (raw HF
            # hidden size, e.g. 768 for xlm-roberta-base) and patch_count
            # (vision grid, e.g. 7x7=49 for ViT-B/32 @ 224px) differ from
            # the original openai/clip-vit-base-patch16 (512, 196), so
            # `resizer`/`clip_sim_resizer` are re-initialized at the new
            # sizes -- everything else (backbone, decoder) still warm-starts
            # from an English checkpoint via strict=False loading.
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                multilingual_clip_model, pretrained=multilingual_clip_pretrained,
                output_dict=False,
            )
            clip_model.visual.output_tokens = True
            clip_model.text.output_tokens = True
            text_dim = clip_model.text.transformer.config.hidden_size
            grid = clip_model.visual.grid_size
            patch_count = grid[0] * grid[1] if isinstance(grid, (tuple, list)) else grid * grid

            self.resizer = FFNLayer(d_model=text_dim, d_out=hidden_dim, dim_feedforward=hidden_dim, dropout=0.1)
            self.clip_sim_resizer = FFNLayer(d_model=patch_count, d_out=hidden_dim, dim_feedforward=hidden_dim, dropout=0.0)
            self.clip_text = clip_model.text
            self.clip_vision = clip_model.visual
            self.tokenizer = open_clip.get_tokenizer(multilingual_clip_model)
            self.processor = clip_preprocess

            for p in self.clip_text.parameters():
                p.requires_grad_(False)
            for p in self.clip_vision.parameters():
                p.requires_grad_(False)

            self.nlp = None  # unused in multilingual mode; see extract_noun_phrase_ko
        else:
            self.resizer = FFNLayer(d_model=512, d_out=hidden_dim, dim_feedforward=hidden_dim, dropout = 0.1)
            self.clip_sim_resizer = FFNLayer(d_model=196, d_out=hidden_dim, dim_feedforward=hidden_dim, dropout=0.0)
            self.clip_text = CLIPTextModelWithProjection.from_pretrained('openai/clip-vit-base-patch16')
            self.tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-base-patch16')
            self.clip_vision = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-base-patch16')
            self.processor = AutoProcessor.from_pretrained('openai/clip-vit-base-patch16')

            for p in self.clip_text.parameters():
                p.requires_grad_(False)
            for p in self.clip_vision.parameters():
                p.requires_grad_(False)

            self.nlp = spacy.load("en_core_web_sm")
    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = VideoHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            num_classes=sem_seg_head.num_classes,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        weight_dict.update({"loss_contrastive": 2})
        losses = ["labels", "masks"]

        criterion = VideoSetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # video
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "multilingual_clip": cfg.MODEL.MULTILINGUAL_CLIP.ENABLED,
            "multilingual_clip_model": cfg.MODEL.MULTILINGUAL_CLIP.MODEL_NAME,
            "multilingual_clip_pretrained": cfg.MODEL.MULTILINGUAL_CLIP.PRETRAINED,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def norm(self, x):
        return x / x.norm(p=2, dim=-1, keepdim=True)


    def forward_clip_text(self, captions):
        if self.multilingual_clip:
            return self._forward_clip_text_ko(captions)
        inputs_clip_text = self.tokenizer(captions, return_tensors = 'pt', padding = True).to(self.device)
        outputs = self.clip_text(**inputs_clip_text)
        text_masks = inputs_clip_text.attention_mask.ne(1).bool()

        text_features = outputs.last_hidden_state 
        text_features = self.resizer(text_features) 
        text_features = NestedTensor(text_features, text_masks) # NestedTensor
        text_pos = self.pe_text(text_features).permute(2, 0, 1)  # [length, batch_size, c]
        text_word_features, text_word_masks = text_features.decompose() 
        text_word_features = text_word_features.permute(1, 0, 2) # [length, batch_size, c] 
        text_sentence_features = text_word_features.mean(0)

        return None, (text_word_features, text_word_masks, text_pos, text_sentence_features)

    def _forward_clip_text_ko(self, captions):
        token_ids = self.tokenizer(captions).to(self.device)
        pad_id = self.tokenizer.tokenizer.pad_token_id
        text_masks = token_ids.eq(pad_id)

        _, text_features = self.clip_text(token_ids)  # (pooled_proj, raw_hidden_states) - output_tokens=True
        text_features = self.resizer(text_features)
        text_features = NestedTensor(text_features, text_masks)
        text_pos = self.pe_text(text_features).permute(2, 0, 1)
        text_word_features, text_word_masks = text_features.decompose()
        text_word_features = text_word_features.permute(1, 0, 2)
        text_sentence_features = text_word_features.mean(0)

        return None, (text_word_features, text_word_masks, text_pos, text_sentence_features)

    def forward_noun_phrase(self, images, noun_phrases, num_frames):
        if self.multilingual_clip:
            return self._forward_noun_phrase_ko(images, noun_phrases, num_frames)
        templeted_sentence = ['a photo of {}'.format(noun_phrase) for noun_phrase in noun_phrases]
        inputs_clip_vision = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs_vision = self.clip_vision(**inputs_clip_vision)
        outputs_patch_vision = self.clip_vision.visual_projection(outputs_vision.last_hidden_state)

        inputs_clip_np = self.tokenizer(templeted_sentence, return_tensors = 'pt', padding = True).to(self.device)
        outputs_np = self.clip_text(**inputs_clip_np)
        outputs_np = outputs_np.text_embeds.unsqueeze(-1)

        
        outputs_patch_vision_ = outputs_patch_vision.view(-1, num_frames, *outputs_patch_vision.shape[1:])
        

        lst = []
        for i in range(num_frames):
            outputs_patch_vision = outputs_patch_vision_[:, i]
            similarity = outputs_patch_vision.repeat(outputs_np.size(0) // outputs_patch_vision.size(0), 1, 1) @ outputs_np # 3B, 197, C x 3B x C x 1 -> 3B x 197 x 1
            list_vis = self.clip_sim_resizer(similarity.transpose(1, 2)[:, :, 1:]) # 3B x 196 x 1
            lst.append(list_vis)
        
        list_vis = torch.stack(lst).mean(0)
        return list_vis.transpose(0, 1)

    _CLIP_KO_IMAGE_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    _CLIP_KO_IMAGE_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    def _preprocess_clip_ko(self, images, size=224):
        # `images` are uint8 [0,255] CHW tensors straight from the dataset
        # mapper (no PIL objects available here), so resize/normalize with
        # plain tensor ops instead of the PIL-oriented open_clip transform.
        batch = torch.stack(images).float() / 255.0
        _, _, h, w = batch.shape
        scale = size / min(h, w)
        new_h, new_w = round(h * scale), round(w * scale)
        batch = F.interpolate(batch, size=(new_h, new_w), mode="bicubic", align_corners=False, antialias=True)
        top = (new_h - size) // 2
        left = (new_w - size) // 2
        batch = batch[:, :, top:top + size, left:left + size]
        mean = self._CLIP_KO_IMAGE_MEAN.to(batch.device, batch.dtype)
        std = self._CLIP_KO_IMAGE_STD.to(batch.device, batch.dtype)
        return (batch - mean) / std

    def _forward_noun_phrase_ko(self, images, noun_phrases, num_frames):
        templeted_sentence = ['{}의 사진'.format(noun_phrase) for noun_phrase in noun_phrases]
        pixel_values = self._preprocess_clip_ko(images)
        _, tokens_vision = self.clip_vision(pixel_values)  # output_tokens=True -> (pooled, patch tokens, no CLS)
        outputs_patch_vision = tokens_vision @ self.clip_vision.proj  # project patches into the shared embed space

        token_ids = self.tokenizer(templeted_sentence).to(self.device)
        pooled_np, _ = self.clip_text(token_ids)
        outputs_np = pooled_np.unsqueeze(-1)

        outputs_patch_vision_ = outputs_patch_vision.view(-1, num_frames, *outputs_patch_vision.shape[1:])

        lst = []
        for i in range(num_frames):
            outputs_patch_vision = outputs_patch_vision_[:, i]
            similarity = outputs_patch_vision.repeat(outputs_np.size(0) // outputs_patch_vision.size(0), 1, 1) @ outputs_np
            list_vis = self.clip_sim_resizer(similarity.transpose(1, 2))  # no CLS to drop here
            lst.append(list_vis)

        list_vis = torch.stack(lst).mean(0)
        return list_vis.transpose(0, 1)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """ 
        images = []
        captions = []

        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
                
            captions.append(video["expression"])
            if self.training:
                captions.append(video["negative_exp"])
                captions.append(video["positive_exp"])
                
        num_frames = len(images) // len(batched_inputs)

        
        prob, captions_feat = self.forward_clip_text(captions)

        #CLIP Prior
        if self.multilingual_clip:
            noun_phrases = [extract_noun_phrase_ko(caption) for caption in captions]
        else:
            noun_phrases = [extract_noun_phrase(caption, self.nlp(caption)) for caption in captions]
        list_vis = self.forward_noun_phrase(images, noun_phrases, num_frames)

        
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        try:
            features = self.backbone(images.tensor)
        except:
            features = self.backbone(images.tensor, num_frames)

        if self.training:    
            for k in features:
                features[k] = features[k][:, None].repeat(1, 3, 1, 1, 1).flatten(0, 1)

    
        
        outputs = self.sem_seg_head(features, captions_feat, list_vis)
        outputs["exps"] = captions
        if self.training:
            # mask classification target
            targets = self.prepare_targets(batched_inputs, images)

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]

            mask_cls_result = mask_cls_results[0]
            # upsample masks
            mask_pred_result = retry_if_cuda_oom(F.interpolate)(
                mask_pred_results[0],
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            sentence_feats = outputs["sentence_feats"]

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation

            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])
            out = retry_if_cuda_oom(self.inference)(mask_cls_result, mask_pred_result, image_size, height, width, sentence_feats)
            out["texts"] = outputs["texts"]

            del outputs
            return out
   
    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        gt_instances = []
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            for i in range(_num_instance):
                mask_shape = [1, self.num_frames, h_pad, w_pad]
                gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

                gt_ids_per_video = []
                for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                    targets_per_frame = targets_per_frame.to(self.device)
                    h, w = targets_per_frame.image_size

                    gt_ids_per_video.append(targets_per_frame.gt_ids[i:i+1, None])
                    gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor[i:i+1]

                gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)
                
                gt_classes_per_video = targets_per_frame.gt_classes[i:i+1]          # N,
                gt_ids_per_video = gt_ids_per_video[i:i+1]                          # N, num_frames

                gt_instances.append({"labels": gt_classes_per_video, "ids": gt_ids_per_video})
                gt_masks_per_video = gt_masks_per_video.float()          # N, num_frames, H, W
                gt_instances[-1].update({"masks": gt_masks_per_video})
            gt_instances.append(gt_instances[-2])
        return gt_instances

    def inference(self, pred_cls, pred_masks, img_size, output_height, output_width, sentence_feats=None):
        if len(pred_cls) > 0:   
            scores = F.softmax(pred_cls[:, :-1], dim=0)
            labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
            
            # keep top-1 predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(1, sorted=False)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.sem_seg_head.num_classes
            
            pred_masks_ori = pred_masks.clone()
            pred_masks = pred_masks[topk_indices]
            pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]
            
            pred_masks = F.interpolate(
                pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
            )
            masks = pred_masks > 0.

            out_scores = scores_per_image.tolist()
            out_labels = labels_per_image.tolist()
            out_masks = [m for m in masks.cpu()]
        else:
            out_scores = []
            out_labels = []
            out_masks = []

        video_output = {
            "image_size": (output_height, output_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
            "pred_ids": topk_indices,
            "sentence_feats": sentence_feats
        }

        return video_output
