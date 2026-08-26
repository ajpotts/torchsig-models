# GNU Radio Conference 2026 XCiT SigMF CTF

This challenge for [GNU Radio Conference 2026](https://events.gnuradio.org/event/28/)
provides five unlabeled SigMF captures. Participants use the
official TorchSig Models v1.0.0 narrowband XCiT checkpoint to classify them and
join the first letter of each prediction.

Open `solve_challenge.ipynb` from the repository root or this directory. The
notebook downloads the released checkpoint on first use.

Challenge maintainers can reproduce the captures in a GNU Radio Python
environment containing TorchSig 2.1.x:

```bash
python generate_captures.py --output-dir captures
```

The script uses TorchSig for seeded signal generation, GNU Radio for writing
the complex sample stream, and SigMF metadata for portable capture files. Its
`--writer numpy` option is intended only for development systems without GNU
Radio.
