# -*- coding: utf-8 -*-
"""
Created on Wed Apr 08 10:15:04 2020

@author: Sven
"""
import pandas as pd


def cleanupdata():
    dyomics_data_raw = pd.read_csv('../raw_data/dyomics.csv')
    dyomics_data_enriched = pd.read_csv('../raw_data/dyomics_enriched.csv')
    dyomics_data_combined = pd.merge(dyomics_data_raw, dyomics_data_enriched, on='Structure')
    dyomics_data = dyomics_data_combined.set_index('Structure')
    dyomics_data.to_csv('../data/dyomics_data.csv')

    IDT_data = pd.read_csv('../raw_data/IDT.csv')

    photochemcad_data_raw = pd.read_csv('../raw_data/photochemcad.csv')
    photochemcad_data_compositions = pd.read_csv('../data/pubChemCompositionRetrival_temp.csv')
    photochemcad_data_enriched = pd.read_csv('../raw_data/photochemcad_enriched.csv')
    photochemcad_data_partial_combined = pd.merge(photochemcad_data_compositions,
                                                  photochemcad_data_raw, on='Structure')
    photochemcad_data_combined = pd.merge(photochemcad_data_partial_combined, photochemcad_data_enriched,
                                          on='Structure')
    photochemcad_data = photochemcad_data_combined.set_index('Structure')
    photochemcad_data.to_csv('../data/photochemcad_data.csv')
    all_data = dyomics_data_combined.append(IDT_data).append(photochemcad_data_combined)
    all_data.set_index('Structure', inplace=True)
    all_data.to_csv('../data/all_data.csv')


if __name__ == "__main__":
    cleanupdata()
