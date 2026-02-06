WPA Python (ported from MATLAB)

Requirements:
  - numpy
  - pillow

Run example:
  python workflow.py --in grass.jpg --out out.png --wa_sel 30 --wa_en 1 --linearize --keepY --warm_end 4500 --tint_k 0.2 --chroma_pullback 0.35

Notes:
  - If you want strict document behavior: use --bin_mode doc_step
  - If you want smooth across nodes: use --bin_mode doc_linear
  - "linearize" is strongly recommended because the CAT derivation assumes linear RGB.
