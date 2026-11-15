import argparse
import collections
import sys

import numpy as np
import torch
import yaml

CHUNK_SAMPLES = 485100


def real_dft_matrices(length):
    positions = torch.arange(length, dtype=torch.float64)
    bins = torch.arange(length // 2 + 1, dtype=torch.float64)
    angles = 2 * torch.pi * bins[:, None] * positions[None, :] / length
    scale = 1.0 / length**0.5
    forward_cosine = (torch.cos(angles) * scale).T.float()
    forward_sine = (-torch.sin(angles) * scale).T.float()
    weights = torch.full((length // 2 + 1,), 2.0, dtype=torch.float64)
    weights[0] = 1.0
    if length % 2 == 0:
        weights[-1] = 1.0
    inverse_cosine = (weights[:, None] * torch.cos(angles) * scale).float()
    inverse_sine = (-weights[:, None] * torch.sin(angles) * scale).float()
    return forward_cosine, forward_sine, inverse_cosine, inverse_sine


def replace_fft_with_matrices(network, frames):
    from models.scnet.separation import FeatureConversion

    forward_cosine, forward_sine, inverse_cosine, inverse_sine = real_dft_matrices(frames)

    def make_forward(module):
        def forward(features):
            if module.inverse:
                real = features[:, : module.channels // 2]
                imaginary = features[:, module.channels // 2 :]
                return real @ inverse_cosine + imaginary @ inverse_sine
            return torch.cat([features @ forward_cosine, features @ forward_sine], dim=1)

        return forward

    for module in network.modules():
        if isinstance(module, FeatureConversion):
            module.forward = make_forward(module)


class ScnetCore(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, spectrum):
        network = self.network
        batch, _, frequency_bins, frames = spectrum.shape
        skips = collections.deque()
        lengths = collections.deque()
        original_lengths = collections.deque()
        features = spectrum
        for encoder_layer in network.encoder:
            features, skip, layer_lengths, layer_original_lengths = encoder_layer(features)
            skips.append(skip)
            lengths.append(layer_lengths)
            original_lengths.append(layer_original_lengths)
        features = network.separation_net(features)
        for fusion_layer, upsample_layer in network.decoder:
            features = fusion_layer(features, skips.pop())
            features = upsample_layer(features, lengths.pop(), original_lengths.pop())
        return features.view(batch, network.dims[0], -1, frequency_bins, frames)


def padded_length(network, samples):
    padding = network.hop_length - samples % network.hop_length
    if (samples + padding) // network.hop_length % 2 == 0:
        padding += network.hop_length
    return padding


def to_spectrum(network, chunk):
    padding = padded_length(network, chunk.shape[-1])
    padded = torch.nn.functional.pad(chunk, (0, padding))
    flat = padded.reshape(-1, padded.shape[-1])
    complex_spectrum = torch.stft(flat, **network.stft_config, return_complex=True)
    real = torch.view_as_real(complex_spectrum)
    batch = chunk.shape[0]
    return real.permute(0, 3, 1, 2).reshape(batch, 4, real.shape[1], real.shape[2]), padding


def from_spectrum(network, features, padding):
    batch, sources, _, frequency_bins, frames = features.shape
    pairs = features.reshape(-1, 2, frequency_bins, frames).permute(0, 2, 3, 1)
    complex_spectrum = torch.view_as_complex(pairs.contiguous())
    waveform = torch.istft(complex_spectrum, **network.stft_config)
    waveform = waveform.reshape(batch, sources, network.audio_channels, -1)
    return waveform[:, :, :, :-padding]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zfturbo-directory", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    sys.path.insert(0, arguments.zfturbo_directory)
    from models.scnet.scnet import SCNet

    with open(arguments.config) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    network = SCNet(**config["model"]).eval()
    network.load_state_dict(torch.load(arguments.checkpoint, map_location="cpu"))
    core = ScnetCore(network).eval()

    chunk = torch.randn(1, 2, CHUNK_SAMPLES) * 0.1
    with torch.no_grad():
        spectrum, padding = to_spectrum(network, chunk)
        original_features = core(spectrum)
        replace_fft_with_matrices(network, spectrum.shape[-1])
        reference_features = core(spectrum)
        matrix_difference = (reference_features - original_features).abs().max().item()
        print(f"matrix_dft_vs_fft max_abs_difference={matrix_difference:.2e}")
        reference_waveform = from_spectrum(network, reference_features, padding)
        original_waveform = network(chunk)
    parity = (reference_waveform - original_waveform).abs().max().item()
    print(f"split_forward_vs_original max_abs_difference={parity:.2e} spectrum_shape={tuple(spectrum.shape)} padding={padding}")

    torch.onnx.export(
        core,
        (spectrum,),
        arguments.output,
        input_names=["spectrum"],
        output_names=["features"],
        opset_version=17,
        dynamo=False,
    )

    import onnxruntime

    session = onnxruntime.InferenceSession(arguments.output, providers=["CPUExecutionProvider"])
    onnx_features = session.run(None, {"spectrum": spectrum.numpy()})[0]
    difference = np.abs(onnx_features - reference_features.numpy()).max()
    print(f"onnx_vs_torch max_abs_difference={difference:.2e} output_shape={onnx_features.shape}")


if __name__ == "__main__":
    main()
