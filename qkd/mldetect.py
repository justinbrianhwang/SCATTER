"""One-class anomaly detector trained on honest telemetry only.

Matches the normal-data-only philosophy of recent DV-QKD intrusion detectors
(Deep SVDD / one-class SVM). The detector never sees attack labels; it learns
the honest telemetry manifold and flags blocks that fall outside it. The
decision threshold is calibrated to a target false-alarm rate (FAR) on held-out
honest blocks, so all attacks are compared at the same operating point.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


class AnomalyDetector:
    def __init__(self, kind: str = "ocsvm", nu: float = 0.05, seed: int = 0):
        self.kind = kind
        self.scaler = StandardScaler()
        if kind == "ocsvm":
            self.model = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
        elif kind == "iforest":
            self.model = IsolationForest(n_estimators=200, random_state=seed,
                                        contamination="auto")
        else:
            raise ValueError(kind)
        self.threshold_ = None

    def fit(self, X_honest: np.ndarray, far: float = 0.01,
            X_cal: np.ndarray | None = None) -> "AnomalyDetector":
        """Fit on honest blocks; set threshold at ``far`` on calibration blocks."""
        Xs = self.scaler.fit_transform(X_honest)
        self.model.fit(Xs)
        cal = X_cal if X_cal is not None else X_honest
        scores = self.anomaly_score(cal)
        # Flag if score exceeds the (1-far) quantile of honest scores.
        self.threshold_ = float(np.quantile(scores, 1.0 - far))
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Higher = more anomalous."""
        Xs = self.scaler.transform(X)
        # decision_function: higher = more normal -> negate.
        return -self.model.decision_function(Xs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Boolean alarm per block at the calibrated threshold."""
        return self.anomaly_score(X) > self.threshold_
