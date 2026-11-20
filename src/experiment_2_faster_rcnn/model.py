"""Faster R-CNN (torchvision) prilagođen DeepPCB-u.

Polazi od modela istreniranog na COCO skupu (91 klasa) i zamjenjuje mu zadnji sloj ("glavu")
tako da predviđa 7 klasa: 0 = pozadina i 1-6 = tipovi defekata (iste oznake kao u DeepPCB-u).
Ostatak mreže (backbone ResNet-50 + FPN i RPN) zadržava COCO težine i samo se dotrenira.
"""
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

NUM_CLASSES = 7  # pozadina + 6 tipova defekata


def build_model(pretrained=True, min_size=800, max_size=1333, score_thresh=0.01):
    """Vraća Faster R-CNN (ResNet-50 FPN v2) sa 7 izlaznih klasa.

    min_size/max_size: model interno skalira sliku tako da kraća strana bude min_size
        (640 -> 800 po defaultu, što pomaže kod sitnih defekata). Boxovi u izlazu su uvijek
        u pikselima originalne slike 640x640.
    score_thresh: predikcije ispod ove pouzdanosti se odbacuju. Namjerno nisko (0.01), jer mAP
        računa cijelu precision-recall krivu; previsok prag bi vještački smanjio mAP.
    """
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        min_size=min_size,
        max_size=max_size,
        box_score_thresh=score_thresh,
        box_detections_per_img=100,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model
