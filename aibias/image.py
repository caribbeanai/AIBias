"""Image bias detection.

For AI-generated imagery the key bias axes are:
- WHO is depicted (skin tone, perceived gender, perceived age) across a set
  of generations for a "neutral" prompt (e.g. "a CEO", "a nurse")
- HOW they are depicted (pose, attire, setting context).

This module covers the *programmatic* signals:
  * Face detection counts (OpenCV Haar / Mediapipe / face_recognition if any)
  * Skin-tone histogram on ITA (Individual Typology Angle) — Monk Skin Tone
    scale buckets.
  * Aggregation across a batch of images (set-level fairness metrics).

For perceived gender / age / ethnicity attributes which require classifiers
that themselves carry bias, this module exposes a clean *plug-in* interface
(`AttributeClassifier`) and a Claude-vision fallback (driven by the SKILL.md).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterable, Protocol, Any

from .metrics import representation_ratio, bias_index


# Monk Skin Tone scale (Google, 2022) approximate ITA boundaries.
# ITA = arctan((L* - 50) / b*) * (180/pi). Larger ITA = lighter skin.
MONK_ITA_BINS = [
    ("MST-1/2 (very light)", 55, 90),
    ("MST-3/4 (light)", 41, 55),
    ("MST-5/6 (medium)", 28, 41),
    ("MST-7/8 (tan/brown)", 10, 28),
    ("MST-9/10 (dark)", -90, 10),
]


@dataclass
class ImageObservation:
    path: str
    n_faces: int
    skin_tone_bucket: str | None
    attributes: dict[str, str] = field(default_factory=dict)  # e.g. {"perceived_gender": "..."}
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ImageBiasReport:
    n_images: int
    n_with_faces: int
    skin_tone_distribution: dict[str, int]
    attribute_distributions: dict[str, dict[str, int]]
    observations: list[ImageObservation]
    axis_scores: dict[str, float]
    bias_index: dict[str, Any]
    findings: list[str]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["observations"] = [o.to_dict() for o in self.observations]
        return d

    def summary(self) -> str:
        lines = [
            f"Images analyzed: {self.n_images} ({self.n_with_faces} with faces)",
            f"Bias Index: {self.bias_index['overall']}/100",
            "Skin-tone distribution: "
            + ", ".join(f"{k}={v}" for k, v in self.skin_tone_distribution.items()),
        ]
        for axis, dist in self.attribute_distributions.items():
            lines.append(f"  {axis}: " + ", ".join(f"{k}={v}" for k, v in dist.items()))
        for f in self.findings:
            lines.append(f"  - {f}")
        return "\n".join(lines)


class AttributeClassifier(Protocol):
    """Optional plug-in: callable mapping (image_path) -> {axis: label}.

    Implementations may wrap DeepFace, FairFace, a Claude vision call, etc.
    They are responsible for declaring their own bias profile.
    """
    def __call__(self, image_path: str) -> dict[str, str]: ...


class ImageBiasDetector:
    def __init__(
        self,
        attribute_classifier: AttributeClassifier | None = None,
        face_detector: Callable[[Any], list] | None = None,
    ):
        self.attribute_classifier = attribute_classifier
        self._face_detector = face_detector or self._load_default_face_detector()

    def analyze(self, image_paths: Iterable[str | Path]) -> ImageBiasReport:
        observations: list[ImageObservation] = []
        skin_dist: Counter = Counter()
        attr_dists: dict[str, Counter] = {}

        for p in image_paths:
            p = str(p)
            obs = self._analyze_one(p)
            observations.append(obs)
            if obs.skin_tone_bucket:
                skin_dist[obs.skin_tone_bucket] += 1
            for axis, label in obs.attributes.items():
                attr_dists.setdefault(axis, Counter())[label] += 1

        axis_scores: dict[str, float] = {}
        findings: list[str] = []

        # Skin-tone imbalance score: total-variation distance from uniform.
        if skin_dist:
            shares = representation_ratio(dict(skin_dist))
            uniform = 1 / len(MONK_ITA_BINS)
            tvd = 0.5 * sum(abs(s - uniform) for s in shares.values())
            axis_scores["skin_tone"] = min(1.0, tvd * 2)
            if tvd > 0.3:
                dom = max(shares, key=shares.get)
                findings.append(
                    f"Skin-tone representation is skewed (TVD={tvd:.2f}); "
                    f"'{dom}' is over-represented at {shares[dom]*100:.0f}%."
                )

        # Attribute axis imbalances
        for axis, counts in attr_dists.items():
            shares = representation_ratio(dict(counts))
            if not shares:
                continue
            mx = max(shares.values())
            if mx >= 0.8 and len(shares) > 1:
                dom = max(shares, key=shares.get)
                findings.append(
                    f"On '{axis}', '{dom}' appears in {mx*100:.0f}% of images."
                )
                axis_scores[axis] = mx
            else:
                # use spread
                axis_scores[axis] = max(shares.values()) - min(shares.values())

        bi = bias_index(axis_scores, sample_size=len(observations))

        return ImageBiasReport(
            n_images=len(observations),
            n_with_faces=sum(1 for o in observations if o.n_faces > 0),
            skin_tone_distribution=dict(skin_dist),
            attribute_distributions={k: dict(v) for k, v in attr_dists.items()},
            observations=observations,
            axis_scores=axis_scores,
            bias_index=bi.to_dict(),
            findings=findings,
        )

    # ----------------------------------------------------------------- helpers
    def _analyze_one(self, path: str) -> ImageObservation:
        notes = []
        n_faces = 0
        skin_bucket = None
        attrs: dict[str, str] = {}

        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
            img = cv2.imread(path)
            if img is None:
                notes.append("could not read image")
                return ImageObservation(path, 0, None, notes=notes)
            faces = self._face_detector(img) if self._face_detector else []
            n_faces = len(faces)
            if faces:
                # Use largest face crop for skin tone estimation.
                fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                crop = img[fy:fy + fh, fx:fx + fw]
                skin_bucket = self._skin_tone_bucket(crop, cv2, np)
            else:
                notes.append("no face detected")
        except ImportError:
            notes.append("opencv-python not installed; programmatic analysis skipped")
        except Exception as e:  # noqa: BLE001
            notes.append(f"analysis error: {e}")

        if self.attribute_classifier:
            try:
                attrs = self.attribute_classifier(path)
            except Exception as e:  # noqa: BLE001
                notes.append(f"classifier error: {e}")

        return ImageObservation(path, n_faces, skin_bucket, attrs, notes)

    def _load_default_face_detector(self):
        try:
            import cv2  # type: ignore
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)

            def detect(img):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                return list(cascade.detectMultiScale(gray, 1.1, 5))

            return detect
        except Exception:
            return None

    def _skin_tone_bucket(self, face_bgr, cv2, np) -> str | None:
        # Convert to L*a*b* and grab central patch (likely cheek) average.
        h, w = face_bgr.shape[:2]
        cy0, cy1 = int(h * 0.55), int(h * 0.75)
        cx0, cx1 = int(w * 0.25), int(w * 0.75)
        patch = face_bgr[cy0:cy1, cx0:cx1]
        if patch.size == 0:
            return None
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
        L = float(np.mean(lab[..., 0])) * 100 / 255  # back to L* in [0,100]
        b = float(np.mean(lab[..., 2])) - 128
        if abs(b) < 1e-6:
            return None
        ita = math.degrees(math.atan2(L - 50, b))
        for name, lo, hi in MONK_ITA_BINS:
            if lo <= ita < hi:
                return name
        return MONK_ITA_BINS[-1][0]
