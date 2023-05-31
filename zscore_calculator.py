#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 12:36:59 2023

@author: christopher
"""


from optparse import OptionParser
from Bio import SeqIO
import RNA as vienna
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from math import factorial
from itertools import combinations, product
import sys
import os

__version__ = "0.1"
__author__ = "Christopher Klapproth"
__institution__= "University Leipzig"

rng = np.random.default_rng()


###############################################
#
#               File handling
#
###############################################


def read_fasta_file(filename):
    handle = SeqIO.parse(open(filename, 'r'), format="fasta")
    records = [f for f in handle]
    return records


def designate_output_prefix(output):
    if len(output) == 0:
        return ""
    elif os.path.isdir(output):
        pass
    else:
        os.mkdir(output.strip("/"))
    return output


def write_tsv(filename, ids, lengths, zs, ps):
    with open(filename, 'w') as f:
        f.write("ID\tlength\tz-score\tp-value\n")
        
        for i in range(0, len(ids)):
            f.write("%s\t%s\t%s\t%s\n" % (ids[i], lengths[i], zs[i], ps[i]))
            
            
def save_energy_histogram(id_, x_mfe, mfes, outdir):
    data = {
        "MFE": mfes,
        }
    id_ = id_.split("/")[0]
    plt.grid(True)
    g = sns.histplot(data=data, x="MFE", edgecolor="k", kde=True, stat="probability")

    
    plt.title("Simulation results for %s" % id_)
    plt.vlines(x=[x_mfe],ymin=0, ymax=1, colors=["r"])
    plt.savefig(os.path.join(outdir, "%s_simulation.png" % id_), dpi=300, bbox_inches="tight")
    plt.clf()


###############################################
#
#               Altschulerickson Algorithm
#
###############################################


def multiset_coefficient(num_elements, cardinality):
    """Return multiset coefficient.

      num_elements      num_elements + cardinality - 1
    ((            )) = (                              )
      cardinality                cardinality
    """
    if num_elements < 1:
        raise ValueError('num_elements has to be an int >= 1')
    ret = factorial(num_elements+cardinality-1)
    ret //= factorial(cardinality)
    ret //= factorial(num_elements-1)
    return ret

def multiset_multiplicities(num_elements, cardinality):
    """Generator function for element multiplicities in a multiset.

    Arguments:
    - num_elements -- number of different elements the multisets are chosen from
    - cardinality -- cardinality of the multisets
    """
    if cardinality < 0:
        raise ValueError('expected cardinality >= 0')
    if num_elements == 1:
        yield (cardinality,)
    elif num_elements > 1:
        for count in range(cardinality+1):
            for other in multiset_multiplicities(num_elements-1,
                                                 cardinality-count):
                yield (count,) + other
    else:
        raise ValueError('expected num_elements >= 1')

def multinomial_coefficient(*args):
    """Return multinomial coefficient.

     sum(k_1,...,k_m)!
    (                 )
      k_1!k_2!...k_m!
    """
    if len(args) == 0:
        raise TypeError('expected at least one argument')
    ret = factorial(sum(args))
    for arg in args:
        ret //= factorial(arg)
    return ret

"""Custom error classes."""

class SequenceTypeError(Exception):
    """Raised if a sequence type is neither RNA or DNA.

    RNA and DNA are string constants defined in zscore/literals.py.
    """
    pass

class DefaultValueError(Exception):
    """Raised if a value in zscore/defaults.py fails a validity check."""
    pass

# Type of nucleic acid.
DNA = 'DNA'
RNA = 'RNA'

# Nucleotide symbols.
A = 'A'
C = 'C'
G = 'G'
T = 'T'
U = 'U'

# Valid nucleotides
nucleotides = ["A", "C", "G", "U", "T", "-"]

# DNA nucleotides.
DNA_NUCLEOTIDES = (A, C, G, T)

# RNA nucleotides.
RNA_NUCLEOTIDES = (A, C, G, U)

# Dinucleotides.
AA = 'AA'
AC = 'AC'
AG = 'AG'
AT = 'AT'
AU = 'AU'
CA = 'CA'
CC = 'CC'
CG = 'CG'
CT = 'CT'
CU = 'CU'
GA = 'GA'
GC = 'GC'
GG = 'GG'
GT = 'GT'
GU = 'GU'
TA = 'TA'
TC = 'TC'
TG = 'TG'
TT = 'TT'
UA = 'UA'
UC = 'UC'
UG = 'UG'
UU = 'UU'

# DNA dinucleotides.
DNA_DINUCLEOTIDES = (AA, AC, AG, AT, CA, CC, CG, CT, GA, GC, GG, GT, TA, TC, TG, TT)

# RNA dinucleotides.
RNA_DINUCLEOTIDES = (AA, AC, AG, AU, CA, CC, CG, CU, GA, GC, GG, GU, UA, UC, UG, UU)

from re import (
    escape,
    findall
)

def doublet_counts(sequence, doublets):
    """Return dict of observed doublet counts in given sequence."""
    counts = dict()
    for d in doublets:
        counts[d] = len(findall('(?={0})'.format(escape(d)), sequence))
    return counts

class ZGraphs():
    """
    Container for valid Z graphs.

    Arguments:
    -- z_vertices: elements in sequence not equal to last element in sequence
    -- last: last element in sequence
    """

    def __init__(self, vertices, last):
        if not (2 <= len(vertices) <= 4):
            raise TypeError('expected sequence of 2, 3 or 4 unique elements '
                            'but got {0} {1}'.format(
                                len(vertices),sorted(list(vertices))))
        if last not in vertices:
            raise ValueError('expected last to be in vertices')
        self.vertices = vertices
        self.last = last
        self.nonlast_vertices = sorted(set(vertices).difference(self.last))
        self.valid = self._create_valid(self.vertices, self.nonlast_vertices,
                                        self.last)

    def _create_valid(self, vertices, nonlast_vertices, last):
        z_graphs = []
        for ends in product(*(vertices for _ in range(len(nonlast_vertices)))):
            z_graph = dict(e for e in zip(nonlast_vertices, ends))
            if self._valid(z_graph, last):
                z_graphs.append(z_graph)
        return tuple(z_graphs)

    def _valid(self, z_graph, last):
        if len(z_graph) > 3:
            raise TypeError('expected z_graph to be a dict representing at '
                            'most 3 last edges')
        connected = True
        for tail, head in z_graph.items():
            if head != last:
                head2 = z_graph.get(head)
                if head2 != last:
                    head3 = z_graph.get(head2)
                    if head3 != last:
                        connected = False
                        break
        return connected

    def check(self, z_graph):
        """Returns true if a Z graph is valid."""
        return z_graph in self.valid


class DoubletGraph:
    """
    Doublet Graph of a sequence.

    A doublet of a sequence is any pair of successive elements in the sequence.
    A doublet graph of a sequence is an ordered directed multigraph.
    In such a graph the vertices are the unique elements of the sequence and
    the directed edges are the doublets in that sequence with direction from
    sequence start to sequence end.

    Arguments:
    -- sequence: string representation of the sequence
    """

    def __init__(self, rng, sequence):
        self.rng = rng
        try:
            self.first = sequence[0]
        except IndexError:
            raise TypeError('expected seqeunce to be a non-empty string')
        self.last = sequence[-1]
        self.edges = tuple(e for e in zip(sequence, sequence[1:]))
        self.vertices = sorted(set(sequence))
        self.z_graphs = ZGraphs(self.vertices, self.last)
        self.nonlast_vertices = self.z_graphs.nonlast_vertices
        self.edgelists = self._create_edgelists(self.vertices, self.edges)
        self.edgemultiplicities = self._create_edgemultiplicites(
            self.vertices, self.edgelists)
        self.possible_z_graphs = self._create_possible_z_graphs()

    def _create_edgelists(self, vertices, edges):
        ret = dict((v,[]) for v in vertices)
        for start, end in edges:
            ret.get(start).append(end)
        for v, l in ret.items():
            ret[v] = tuple(l)
        return ret

    def _create_edgemultiplicites(self, vertices, edgelists):
        ret = dict()
        for v, l in edgelists.items():
            m = dict((v, l.count(v)) for v in vertices)
            ret[v] = m
        return ret

    def _create_possible_z_graphs(self):
        ret = []
        for z in self.z_graphs.valid:
            possible = True
            for v in self.nonlast_vertices:
                if z.get(v) not in self.edgelists.get(v):
                    possible = False
                    break
            if possible:
                ret.append(z)
        return ret

    def num_permutations(self):
        total = 0
        for z in self.possible_z_graphs:
            subtotal = 1
            for v in z:
                last = z.get(v)
                multiplicities = self.edgemultiplicities.get(v).copy()
                multiplicities[last] -= 1
                subtotal *= multinomial_coefficient(*multiplicities.values())
            subtotal *= multinomial_coefficient(
                *self.edgemultiplicities.get(self.last).values())
            total += subtotal
        return total

    def random_z_graph(self):
        while(True):
            ret = dict()
            for start in self.nonlast_vertices:
                end = self.rng.choice(self.edgelists.get(start))
                ret[start] = end
            if self.z_graphs.check(ret):
                return ret


class DPPermutations:
    """
    Doublet preserving permutations of a sequence.

    The cardinality of the alphabet must be <= 4 as is the case in DNA or RNA.

    Arguments:
    -- sequence: string representation of the sequence.
    """

    def __init__(self,
                 rng,
                 sequence,
                 sequence_type,
                 sequence_length,
                 di_features):
        self.rng = rng
        self.sequence = "".join([x for x in sequence.upper() if x in nucleotides])
        self.length = sequence_length
        self.sequence_type = sequence_type
        if self.sequence_type == RNA:
            self.DINUCLEOTIDES = RNA_DINUCLEOTIDES
        elif self.sequence_type == DNA:
            self.DINUCLEOTIDES = DNA_DINUCLEOTIDES
        else:
            raise SequenceTypeError
        if self.length <= 3 or len(set(sequence)) <= 1:
            self.trivial = True
        else:
            self.trivial = False
            self.g = DoubletGraph(rng=self.rng, sequence=self.sequence)
            if di_features == None:
                self.dcounts = doublet_counts(self.sequence, self.DINUCLEOTIDES)
            else:
                self.dcounts = di_features

    def num_permutations(self):
        return 1 if self.trivial else self.g.num_permutations()
            
    def shuffle(self):
        if self.trivial:
            return self.sequence
        else:
            last_edges = self.g.random_z_graph()
            edgelists = dict((v, list(l)) for v,l in self.g.edgelists.items())
            for start, end in last_edges.items():
                edgelists.get(start).remove(end)
            for v in self.g.vertices:
                self.rng.shuffle(edgelists.get(v))
            for start, end in last_edges.items():
                edgelists.get(start).append(end)
            ret = self.g.first
            for _ in range(self.length - 1):
                ret += edgelists.get(ret[-1]).pop(0)
            if self.dcounts != doublet_counts(ret, self.DINUCLEOTIDES):
                raise RuntimeError(
                    'Something went very wrong! '
                    'Shuffled sequence is not doublet preserving.')
            return ret


###############################################
#
#               Calculate MFE structures and z-score
#
###############################################


def calc_z_score(x, std, mu):
    return round((x - mu) / std, 4)


def get_distribution(mfes):
    std = np.std(mfes)
    mu = np.mean(mfes)
    return mu, std


def fold_single(seq):
    fc = vienna.fold_compound(seq)
    mfe_structure, mfe = fc.mfe()
    return mfe_structure, mfe


def shuffle_me(seq, samples, sequence_type):
    generator = DPPermutations(sequence=seq, rng=rng, 
                sequence_type=sequence_type, sequence_length=len(seq), di_features=None)
    references = []
    for i in range(0, samples):
        references.append(generator.shuffle())
    
    return references


def z_score_of_MFE(id_, seq, samples, outdir):
    if "U" in seq:
        sequence_type = "RNA"
    else:
        sequence_type = "DNA"
        
    structure, mfe = fold_single(seq)
    
    references = shuffle_me(seq, samples, sequence_type)
    mfes = [fold_single(s)[1] for s in references]
    mu, std = get_distribution(mfes)
    z = calc_z_score(mfe, std, mu)
    p = round(stats.norm.sf(abs(z)), 4)
    
    save_energy_histogram(id_, mfe, mfes, outdir)
    
    return z, p, mfe, structure


###############################################
#
#               Main function
#
###############################################


def main():
    if len(sys.argv) < 2:
        print("Usage:\npython zscore_calculator.py [options] [Fasta 1] [Fasta 2] ...")
        sys.exit()
    usage = "\npython %prog  [options] [Fasta 1] [Fasta 2] ..."
    parser = OptionParser(usage, version="%prog " + __version__)
    parser.add_option("-s","--samples",action="store", type="int", dest="samples", default=1000, help="How many sequences should be simulated to estimate background distribution? Lower numbers should speed up the process but also lower accuracy (Default: 1000).")
    parser.add_option("-p","--plotting",action="store", dest="plotting", default=True, help="Pass '--plotting False' if you do not want a drawing of each individual energy distribution.")
    parser.add_option("-o","--output",action="store",type="string", dest="out_dir", default="", help="Designate a directory for the output. If it does not already exist, it will be created. If nothing is specified, files are written in current working directory.")
    
    options, args = parser.parse_args()
    outdir = designate_output_prefix(options.out_dir)
    n_samples = options.samples
    plotting = options.plotting
    
    ids = []
    lengths = []
    zs = []
    ps = []
    
    for filename in args:
        fasta_records = read_fasta_file(filename)
        for record in fasta_records:
            id_ = str(record.id)
            seq = str(record.seq).replace("-", "").replace("_", "")
            z, p, mfe, structure = z_score_of_MFE(id_, seq, n_samples, outdir)
            
            ids.append(id_)
            lengths.append(len(seq))
            zs.append(z)
            ps.append(p)
            
            print("ID: %s" % id_)
            print("Sequence: %s" % seq)
            print("Structure: %s \n" % structure)
            print("MFE: %s \t z-score of MFE: %s \t p-value: %s" % (mfe, z, p))
            print("\n \n")
    
    write_tsv(os.path.join(outdir, "z_score_summary.tsv"), ids, lengths, zs, ps)


if __name__ == "__main__":
    main()
