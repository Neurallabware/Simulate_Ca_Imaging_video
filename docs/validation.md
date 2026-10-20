# Validation and limitations

The guide and examples were checked against the recorded source snapshot. Documentation uses a separate CPU-only Sphinx build and does not import the scientific runtime, download datasets or weights, execute notebooks, train models, or connect to instruments.

Successful HTML generation does not validate scientific outputs. For a scientific run, record source commit, environment, data/checkpoint identity, resolved configuration, hardware, command, output directory, and the relevant comparison/QC evidence.

Set the random seed, session count, frame count, motion options, and output directory explicitly. MATLAB and Python return orders differ; do not copy MATLAB tuple unpacking into Python.
