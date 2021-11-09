from __future__ import print_function, division

import csv
import functools
import json
from tqdm import tqdm
import os
import random
import warnings

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.
    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------
        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array
        Parameters
        ----------
        distance: np.array shape n-d array
          A distance matrix of any shape
        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.
    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.
    Parameters
    ----------
    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)

max_num_nbr=16
radius=9 
dmin=0
step=0.2
                 

root_dir = 'training/sample'
atom_init_file = os.path.join(root_dir, 'atom_init.json')
ari = AtomCustomJSONInitializer(atom_init_file)
gdf = GaussianDistance(dmin=dmin, dmax=radius, step=step)


root_dir = 'training/sample'

existed = []

f=open('%s/id_prop.csv'%(root_dir))
length = len(f.readlines())
f.close()
f=open('%s/id_prop.csv'%(root_dir))
for j in tqdm(range(length)):
    cif_id, _, target = f.readline().split(',')
    if cif_id in existed:
        continue
    if os.path.isfile('%s/%s.npy'%(root_dir, cif_id)):
        continue
    existed.append(cif_id)
    crystal = Structure.from_file(os.path.join(root_dir,cif_id+'.cif'))
    atom_fea = np.vstack([ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])
#    atom_fea = torch.Tensor(atom_fea)
    all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    nbr_fea_idx, nbr_fea = [], []
    for nbr in all_nbrs:
        if len(nbr) < max_num_nbr:
            warnings.warn('{} not find enough neighbors to build graph. '
                            'If it happens frequently, consider increase '
                            'radius.'.format(cif_id))
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (max_num_nbr - len(nbr)))
            nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [radius + 1.] * (max_num_nbr -
                                                     len(nbr)))
        else:
            nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:max_num_nbr])))
            nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:max_num_nbr])))
    nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
    nbr_fea = gdf.expand(nbr_fea)

    graph_dict = {'atom_fea':atom_fea, 'nbr_fea':nbr_fea, 'nbr_fea_idx':nbr_fea_idx}
     
    np.save('%s/%s'%(root_dir, cif_id), graph_dict)
#    atom_fea = torch.Tensor(atom_fea)

#    nbr_fea = torch.Tensor(nbr_fea)
#    nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
#    target = torch.Tensor([float(target)])
#        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id
