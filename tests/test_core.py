"""Regression coverage for inference, quantization, training, and compression."""

import copy

import numpy as np
import pytest
import torch

import neurofield as nf


@pytest.mark.parametrize("chunk_size", [None, 0, -1, 1, 7, 100])
def test_chunked_inference_preserves_shape_and_disables_gradients(chunk_size):
    torch.manual_seed(0)
    model = nf.MLP(2, 3, hidden_features=8)
    coordinates = torch.rand(3, 5, 2, requires_grad=True)
    expected = model(coordinates).detach()

    output = nf.chunked_inference(
        model, coordinates, chunk_size=chunk_size, device="cpu"
    )

    torch.testing.assert_close(output, expected)
    assert output.shape == (3, 5, 3)
    assert not output.requires_grad
    assert not model.training


@pytest.mark.parametrize(
    ("quant_max", "dtype"),
    [(127, torch.int8), (128, torch.int16), (32768, torch.int32), (2**31, torch.int64)],
)
def test_quantized_parameters_round_trip_and_remain_trainable(quant_max, dtype):
    torch.manual_seed(0)
    model = nf.MLP(2, 3, hidden_features=8)
    restored = copy.deepcopy(model)
    coordinates = torch.rand(5, 2)

    quantized, codes = nf.uniform_quantize(model, quant_max)
    assert quantized is model
    assert (
        nf.uniform_dequantize(restored, nf.decode_dict(nf.encode_dict(codes)))
        is restored
    )
    for entry in codes.values():
        assert entry["quant"].dtype == dtype
        assert entry["quant"].device.type == "cpu"
        assert not entry["scale"].requires_grad

    torch.testing.assert_close(
        restored(coordinates), model(coordinates), rtol=0, atol=0
    )
    model(coordinates).square().mean().backward()
    assert all(torch.isfinite(param.grad).all() for param in model.parameters())


@pytest.mark.parametrize("quantize", [False, True])
def test_image_training_evaluation_and_checkpoint(tmp_path, quantize):
    torch.manual_seed(0)
    image = torch.arange(3 * 8 * 8, dtype=torch.uint8).reshape(3, 8, 8)
    dataset = nf.ImageCoordinateDataset(image, item_id="image")
    model = nf.MLP(2, 3, hidden_features=8)

    results = nf.train(
        model,
        dataset,
        num_epochs=6,
        lr=0.05,
        quantize=quantize,
        quant_interval=2,
        eval_interval=3,
        metrics={"psnr": nf.psnr},
        chunk_size=7,
        log_dir=tmp_path,
        device="cpu",
    )

    history = results["history"]
    assert len(history) == 6
    assert history[-1]["mse"] < history[0]["mse"]
    assert np.isfinite(history[-1]["eval"]["psnr"])
    assert (tmp_path / "image_reconstructed.png").exists()

    restored = nf.MLP(2, 3, hidden_features=8)
    restored.load_state_dict(torch.load(tmp_path / "checkpoint.pt", weights_only=True))
    torch.testing.assert_close(
        restored(dataset.input), model(dataset.input), rtol=0, atol=0
    )
    if quantize:
        nf.uniform_dequantize(restored, results["model_dict"])
        torch.testing.assert_close(
            restored(dataset.input), model(dataset.input), rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_dip_training_keeps_data_on_device(device):
    torch.manual_seed(0)
    image = torch.randint(256, (3, 8, 8), dtype=torch.uint8)
    dataset = nf.DIPImageDataset(image, noise_shape=(4, 16, 16))
    model = nf.DIPSkip(
        4, 3, down_channels=[4, 4], up_channels=[4, 4], skip_channels=[2, 2]
    )

    def loss_fn(batch, model):
        target = batch["target"]
        output = torch.nn.functional.interpolate(
            model(batch["input"]),
            size=target.shape[-2:],
            mode="bicubic",
            antialias=True,
        )
        mse = torch.nn.functional.mse_loss(output, target)
        return mse, {"mse": mse.item()}

    results = nf.train(
        model, dataset, loss_fn=loss_fn, num_epochs=4, eval_interval=-1, device=device
    )

    assert len(results["history"]) == 4
    assert dataset.input.device.type == dataset.target.device.type == device
    assert dataset.original.device.type == "cpu"


def test_arithmetic_coding_preserves_bitstream():
    # This stream exercises unequal alphabets and deferred bits.
    pmfs = [[0.7, 0.3], [0.1, 0.2, 0.7], [0.4, 0.6], [0.5, 0.5]]
    symbols = [0, 2, 1, 0]
    encoded = nf.compression.encode_arithmetic(symbols, pmfs)
    assert encoded == bytes.fromhex("0000000470")
    np.testing.assert_array_equal(
        nf.compression.decode_arithmetic(encoded, pmfs), symbols
    )


def test_lossless_image_compression_round_trip():
    image = torch.arange(3 * 8 * 8, dtype=torch.uint8).reshape(3, 8, 8)
    results = nf.compression.evaluate_compression(
        image,
        nf.compression.encode_pil,
        nf.compression.decode_pil,
        format="PNG",
        return_reconstruction=True,
    )
    assert torch.equal(results["reconstructed"], image)
    assert results["bit rate (bpp)"] > 0
