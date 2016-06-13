from __future__ import absolute_import, division, print_function

import numpy as np

from geminals.hort import ap1rog, hartreefock


def test_hartreefock():

    nelec = 2
    hf_dict = hartreefock(fn="test/h2.xyz", basis="sto-3g", nelec=nelec)
    e_hf, H, G, nuc_nuc = hf_dict["energy"], hf_dict["H"], hf_dict["G"], hf_dict["nuc_nuc"]

    assert isinstance(e_hf, float)
    assert isinstance(nuc_nuc, float)
    assert isinstance(H, np.ndarray)
    assert isinstance(G, np.ndarray)
    assert np.all(np.array(H.shape + G.shape) == H.shape[0])


def test_ap1rog():

    nelec = 2
    ap1rog_dict = ap1rog(fn="test/h2.xyz", basis="sto-3g", nelec=nelec)
    e_ap1rog, x, C = ap1rog_dict["energy"], ap1rog_dict["x"], ap1rog_dict["C"]

    assert isinstance(e_ap1rog, float)
    assert isinstance(x, np.ndarray)
    assert isinstance(C, np.ndarray)
    assert np.all(x == C.ravel())
    e_hf = hartreefock(fn="test/h2.xyz", basis="sto-3g", nelec=nelec)["energy"]
    assert e_ap1rog < e_hf