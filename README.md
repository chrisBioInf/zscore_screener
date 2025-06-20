# Z-Score Calculator for RNA/DNA MFE Structures

A Python tool that calculates Z-scores for minimum free energy (MFE) structures of RNA/DNA sequences using doublet-preserving shuffling.

## Required Libraries

- **Biopython** (`Bio.SeqIO`)
- **ViennaRNA** (`RNA`)  
- **NumPy**
- **SciPy** 
- **Matplotlib**
- **Seaborn**

## Usage

```bash
python zscore_calculator.py [options] [Fasta 1] [Fasta 2] ...
```

## Arguments

- **Positional**: One or more FASTA files containing sequences
- `-s, --samples`: Number of shuffled sequences for background distribution (default: 1000)
- `-p, --plotting`: Generate energy distribution plots (default: True)  
- `-o, --output`: Output directory path (default: current directory)

## Output

- TSV file with Z-scores and p-values
- Energy distribution histograms (PNG files)
- Console output with sequence details and statistics
