"""Fast regression tests for evidence and preprocessing boundaries."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.audit_data import build_report
from src import config
from src.api import compute_research_report
from src.dataset import find_mask_path, preprocess_image


class PreprocessingTests(unittest.TestCase):
    def test_each_strategy_returns_model_dimensions(self):
        image = np.full((80, 140, 3), 96, dtype=np.uint8)
        for strategy in ("roi_crop", "center_crop", "direct_resize"):
            with self.subTest(strategy=strategy):
                result = preprocess_image(image, use_clahe=False, crop_strategy=strategy)
                self.assertEqual(result.shape, (config.IMG_SIZE, config.IMG_SIZE, 3))

    def test_mask_discovery_supports_dataset_suffixes(self):
        with tempfile.TemporaryDirectory() as folder:
            image_path = Path(folder) / "case001.png"
            mask_path = Path(folder) / "case001_tumor.png"
            cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))
            cv2.imwrite(str(mask_path), np.ones((8, 8), dtype=np.uint8))
            self.assertEqual(find_mask_path(str(image_path)), str(mask_path))


class EvidenceBoundaryTests(unittest.TestCase):
    def test_current_data_audit_is_explicitly_provisional(self):
        report = build_report(Path(config.DATA_DIR))
        self.assertFalse(report["audit_passed"])
        self.assertIn("busi", report["failed_datasets"])

    def test_report_never_recommends_clinical_action(self):
        report = compute_research_report(
            "malignant", 0.9, {}, {"dominant_type": "speckle"}, "efficientnet_b0"
        )
        self.assertEqual(report["status"], "Research output only")
        self.assertIn("No clinical action", report["action"])


if __name__ == "__main__":
    unittest.main()
