import argparse
import time

import numpy as np
import onnxruntime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--chunks", type=int, default=10)
    parser.add_argument("--lean-memory", action="store_true")
    parser.add_argument("--chunk-samples", type=int, default=485100)
    arguments = parser.parse_args()

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = arguments.threads
    if arguments.lean_memory:
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
    session = onnxruntime.InferenceSession(arguments.model, options, providers=["CPUExecutionProvider"])
    frames = session.get_inputs()[0].shape[3]
    spectrum = (np.random.randn(1, 4, 2049, frames) * 0.01).astype(np.float32)
    session.run(None, {"spectrum": spectrum})
    started = time.time()
    for _ in range(arguments.chunks):
        session.run(None, {"spectrum": spectrum})
    per_chunk = (time.time() - started) / arguments.chunks
    chunk_seconds = arguments.chunk_samples / 44100
    for overlap in (1, 2, 4):
        chunks_per_minute = 60 / (chunk_seconds / overlap)
        print(f"overlap={overlap} seconds_per_chunk={per_chunk:.2f} estimated_seconds_per_minute={per_chunk * chunks_per_minute:.1f}")


if __name__ == "__main__":
    main()
