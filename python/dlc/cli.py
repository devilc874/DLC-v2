"""
DLC Command-Line Interface.

Usage:
    dlc compress  -i input.bin -o output.dlc [--precision 12] [--chunk-size 100000] [--workers 4]
    dlc decompress -i input.dlc -o output.bin
"""

import argparse
import sys
import numpy as np

from dlc.codec import compress, decompress


def main():
    parser = argparse.ArgumentParser(
        prog="dlc",
        description="DLC (Delta-Linear Compression) for time-series sensor data",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── compress ─────────────────────────────────────────────────────────────
    cp = sub.add_parser("compress", help="Compress a raw float64 binary file")
    cp.add_argument("-i", "--input", required=True, help="Input .bin file (raw float64 LE)")
    cp.add_argument("-o", "--output", required=True, help="Output .dlc file")
    cp.add_argument("--precision", type=int, default=12, help="Quantization bits (default 12)")
    cp.add_argument("--chunk-size", type=int, default=100_000, help="Block size (default 100000)")
    cp.add_argument("--workers", type=int, default=None, help="Thread count (default auto)")

    # ── decompress ───────────────────────────────────────────────────────────
    dp = sub.add_parser("decompress", help="Decompress a .dlc file")
    dp.add_argument("-i", "--input", required=True, help="Input .dlc file")
    dp.add_argument("-o", "--output", required=True, help="Output .bin file (raw float64 LE)")

    args = parser.parse_args()

    if args.command == "compress":
        # Read raw float64 binary (little-endian)
        data = np.fromfile(args.input, dtype='<f8')
        print(f"Read {len(data)} samples from {args.input}")

        compress(
            data, args.output,
            precision_bits=args.precision,
            chunk_size=args.chunk_size,
            num_workers=args.workers,
        )
        print(f"Compressed to {args.output}")

    elif args.command == "decompress":
        result = decompress(args.input)
        result.astype('<f8').tofile(args.output)
        print(f"Decompressed {len(result)} samples to {args.output}")


if __name__ == "__main__":
    main()
