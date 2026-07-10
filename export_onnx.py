import argparse
import logging
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn


class SquareNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 13),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def load_model(weights_path: Path) -> nn.Module:
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    model = SquareNet()
    state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_model(model: nn.Module, output_path: Path, input_shape: tuple[int, int, int, int]) -> None:
    dummy_input = torch.randn(*input_shape, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )


def validate_onnx_model(output_path: Path) -> None:
    exported_model = onnx.load(str(output_path))
    onnx.checker.check_model(exported_model)


def compare_predictions(model: nn.Module, output_path: Path, num_samples: int, seed: int) -> float:
    torch.manual_seed(seed)
    samples = torch.randn(num_samples, 3, 100, 100, dtype=torch.float32)

    with torch.inference_mode():
        torch_output = model(samples).cpu().numpy()

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: samples.numpy().astype(np.float32)})[0]

    max_abs_diff = float(np.max(np.abs(torch_output - onnx_output)))
    logging.info("Maximum absolute difference across %s samples: %.8f", num_samples, max_abs_diff)
    return max_abs_diff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SquareNet PyTorch checkpoint to ONNX")
    parser.add_argument("--weights", type=Path, default=Path("model_best.pth"), help="Path to .pth checkpoint")
    parser.add_argument("--output", type=Path, default=Path("model.onnx"), help="Path to output ONNX model")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy input batch size for export")
    parser.add_argument("--compare-samples", type=int, default=8, help="Number of random samples for numerical comparison")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for comparison")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    model = load_model(args.weights)
    export_model(model, args.output, (args.batch_size, 3, 100, 100))
    validate_onnx_model(args.output)
    logging.info("ONNX model exported and validated at %s", args.output)

    max_abs_diff = compare_predictions(model, args.output, args.compare_samples, args.seed)
    logging.info("Export completed. Maximum absolute prediction difference: %.8f", max_abs_diff)


if __name__ == "__main__":
    main()
