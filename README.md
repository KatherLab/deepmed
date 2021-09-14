# Welcome to Direct End-to-End Pipeline for Medical Imaging

## What is this?

This is an open source platform for end-to-end artificial intelligence (AI) in
computational pathology.  It will enable you to use AI for prediction of any
"label" directly from digitized pathology slides.  Common use cases which can be
reproduced by this pipeline are:

  - prediction of microsatellite instability in colorectal cancer (Kather et
    al., Nat Med 2019)
  - prediction of mutations in lung cancer (Coudray et al., Nat Med 2018)
  - prediction of subtypes of renal cell carcinoma (Lu et al., Nat Biomed Eng
    2021)
  - other possible use cases are summarized by Echle et al., Br J Cancer 2021:
    https://www.nature.com/articles/s41416-020-01122-x

This pipeline is modular, which means that new methods for pre-/postprocessing
or new AI methods can be easily integrated.


## Installation

Deepmed has been tested on both Windows Server 2019 and Ubuntu 20.04.  It
requires a CUDA-enabled NVIDIA GPU and a Python installation of at least version
3.8.  In most cases, deepmed can then be installed by typing:

```bash
pip install git+https://github.com/KatherLab/deepmed
```

In some cases it may be necessary to install pytorch manually in order for it to
recognize the system's GPU.  To do so, please refer to the [pytorch installation
guide].

[pytorch installation guide]: https://pytorch.org/get-started/locally/


## Documentation

To build the project's documentation, we need to install a few more
dependencies:

```bash
pip install sphinx sphinx_rtd_theme
```

After that, we can build the documentation by invoking the `Makefile` or
`make.bat` in the docs dictory, i.e.:

```bash
make -C path/to/deepmed/docs html
```

on Linux systems or

```powershell
path\to\deepmed\docs\make.bat html
```

on Windows.  Afterwards, the documentation can be found in
`docs/build/html/index.html`.


## Tests

Deepmed comes with a set of integration tests.  These can be invoked by running

```bash
cd path/to/deepmed && python -m unittest
```
