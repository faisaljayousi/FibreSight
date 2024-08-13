# FibreSight - Characterisation of Fibres in 2D-Images
[![License](https://img.shields.io/github/license/faisaljayousi/FibreSight)](https://opensource.org/licenses/MIT)
[![Last Commit](https://img.shields.io/github/last-commit/faisaljayousi/FibreSight)](https://github.com/faisaljayousi/FibreSight/commits)

**FibreSight** is a Python module designed for advanced analysis and characterisation of fibrous structures in 2D images. This module builds upon the methodologies described in the following paper:

- Jayousi, F., Descombes, X., Bouilhol, E., Sudaka, A., Van Obberghen-Schilling, E., & Blanc-Féraud, L. (2024). *Detection and Characterisation of Fibronectin Structures in the Tumour Extracellular Matrix*. In *32nd European Signal Processing Conference (EUSIPCO) 2024*, Lyon, France, August 26-30, 2024.

This module offers the following functionalities:
- **Image Enhancement**: Enhancement of images using Gabor filters.
- **Graph Representation**: Extraction of a graph representation of the underlying fibres.
- **Geometric Partitionning**: Application of custom partition methods to segment images into geometrically homogeneous regions.

It is designed to be flexible and extensible, allowing users to integrate their own custom image processing and partitioning techniques.

Please [cite this paper](#how-to-cite) if you use this code in your research.

## Installation

You can install this module using either of the following methods:

### Using pip
Coming soon.

### GitHub
```bash
pip install git+https://github.com/faisaljayousi/FibreSight.git
```

This will install the module along with its dependencies as specified in the `setup.py` file of the repository.

## Usage

### Jupyter Notebook

For a comprehensive guide on using `FibreSight`, including detailed examples and visualisations, refer to the Jupyter Notebook provided in this repository. The notebook demonstrates various functionalities, including image enhancement, graph creation, and partitioning methods.

You can find the notebook here: [FibreSight Usage Notebook](examples/fibre_enhancement.ipynb).


### Using Docker

Coming soon.

## How to cite

```bibtex
@inproceedings{jayousi2024,
  author    = {Faisal Jayousi and Xavier Descombes and Emmanuel Bouilhol and Anne Sudaka and Ellen Van Obberghen-Schilling and Laure Blanc-Féraud},
  title     = {Detection and Characterisation of Fibronectin Structures in the Tumour Extracellular Matrix},
  booktitle = {32nd European Signal Processing Conference, {EUSIPCO} 2024, Lyon, France, August 26-30, 2024},
  pages     = {},
  year      = {2024},
  doi       = {}
}
```

## Acknowledgements

This work was supported by the French government through the [3IA Côte d’Azur](https://3ia.univ-cotedazur.eu/) Investments ANR-19-P3IA-0002, managed by the National Research Agency, Unicancer (TOPNIVO-ORL09) and the Fondation ARC-Unicancer (TOP’UP).
