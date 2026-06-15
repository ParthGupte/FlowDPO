import torch
import torchmetrics
from metrics.flip_detector.model import TinyMNISTBinaryCNN


class FlipRatioMetric(torchmetrics.Metric):
    full_state_update = False

    def __init__(
        self,
        classifier_cls=TinyMNISTBinaryCNN,
        checkpoint_path="src/metrics/flip_detector/saved_checkpoint/tiny_mnist_flip_classifier.pt",
        threshold=0.5,
    ):
        super().__init__(dist_sync_on_step=False)

        self.threshold = threshold

        # Not registered as metric state
        self.classifier = classifier_cls()

        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        self.classifier.load_state_dict(state_dict)
        self.classifier.eval()

        for p in self.classifier.parameters():
            p.requires_grad_(False)

        self.add_state(
            "num_flipped",
            default=torch.tensor(0),
            dist_reduce_fx="sum",
        )

        self.add_state(
            "num_total",
            default=torch.tensor(0),
            dist_reduce_fx="sum",
        )

    @torch.no_grad()
    def update(self, images):
        """
        images: (B,1,28,28)
        """

        device = images.device
        self.classifier = self.classifier.to(device)

        logits = self.classifier(images).squeeze(1)

        probs = torch.sigmoid(logits)

        preds = probs > self.threshold

        self.num_flipped += preds.sum()
        self.num_total += preds.numel()

    def compute(self):
        flipped = self.num_flipped.float()
        total = self.num_total.float()

        unflipped = total - flipped

        return {
            "flip_ratio": flipped / total,
            "unflip_ratio": unflipped / total,
            "num_flipped": flipped,
            "num_unflipped": unflipped,
        }

    def persistent(self, mode=False):
        # prevents classifier from ending up in checkpoints
        return super().persistent(mode)