# -*- coding: utf-8 -*-

# Copyright 2017 Novo Nordisk Foundation Center for Biosustainability,
# Technical University of Denmark.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Supporting functions for syntax checks."""

from __future__ import absolute_import

import logging
import re
from os.path import join, dirname
from builtins import dict

import pandas as pd

from memote.support.helpers import (
    build_reaction_pattern_map,
    find_atp_adp_converting_reactions,
    find_transport_reactions)

LOGGER = logging.getLogger(__name__)

COMPARTMENTS = pd.read_csv(
    join(dirname(__file__), "data", "compartments.csv"),
    header=0, quotechar='"', comment="#")
REACTION_SUFFIX = pd.read_csv(
    join(dirname(__file__), "data", "reaction_suffixes.csv"),
    header=0, quotechar='"', comment="#")
COMPARTMENT_SUFFIX = dict(
    COMPARTMENTS[["symbol", "suffix"]].itertuples(index=False))


def find_mistagged_reaction_compartment(model, reaction_match=None,
        transport_reactions=None):
    """
    Find incorrectly tagged reaction IDs.

    Parameters
    ----------
    model : cobra.Model
        A cobrapy metabolic model.
    reaction_match : dict, optional
        Reactions and their regexp matches.
    transport_reactions : iterable, optional
        A list of transport reactions to be ignored.

    Returns
    -------
    list
        Reactions whose ID suffix does not match their compartment.
    """
    if reaction_match is None:
        reaction_match = build_reaction_pattern_map(model)
    if transport_reactions is None:
        transport_reactions = find_transport_reactions(model)
    wrong = list()
    for rxn in set(reaction_match) - set(transport_reactions):
        match = reaction_match[rxn]
        if len(rxn.compartments) > 1:
            LOGGER.warn("%s spans different compartments.", rxn.id)
        if match is None:
            LOGGER.debug("%s did not match regexp", rxn.id)
            continue
        if set(['c']) == rxn.compartments:
            # cytosolic reaction
            continue
        suffix = match.group("suffix")
        if any(comp in suffix for comp in rxn.compartments):
            LOGGER.debug("%s's compartments (%s) properly tagged", rxn.id,
                         ", ".join(rxn.compartments))
            continue
        wrong.append(rxn)
    return wrong

def find_rxn_id_compartment_suffix(model, suffix):
    """
    Find incorrectly tagged IDs.

    Find non-transport reactions with metabolites in the given compartment
    whose ID is not tagged accordingly.

    Parameters
    ----------
    model : cobra.Model
        A cobrapy metabolic model.
    suffix : str
        The suffix of the compartment to be checked.

    Returns
    -------
    list
        Non-transport reactions that have at least one metabolite in the
        compartment given by `suffix` but whose IDs do not have
        the `suffix` appended.
    """
    transport_rxns = find_transport_reactions(model)
    exchange_demand_rxns = find_demand_and_exchange_reactions(model)

    comp_pattern = re.compile(
        "[A-Z0-9]+\w*?{}\w*?".format(SUFFIX_MAP[suffix])
    )

    rxns = []
    for rxn in model.reactions:
        if suffix in rxn.compartments:
            if ('biomass' not in rxn.id.lower()) and (
                    rxn not in transport_rxns and
                    rxn not in exchange_demand_rxns):
                rxns.append(rxn)

    return [rxn for rxn in rxns if not comp_pattern.match(rxn.id)]


def find_rxn_id_suffix_compartment(model, suffix):
    """
    Find incorrectly tagged non-transport reactions.

    Find non-transport reactions whose ID bear a compartment tag but which do
    not contain any metabolites in the given compartment.

    Parameters
    ----------
    model : cobra.Model
        A cobrapy metabolic model.
    suffix : str
        The suffix of the compartment to be checked.

    Returns
    -------
    list
        Non-transport reactions that have at least one metabolite in the
        compartment given by `suffix` but whose IDs do not have
        the `suffix` appended.
    """
    transport_rxns = find_transport_reactions(model)
    exchange_demand_rxns = find_demand_and_exchange_reactions(model)

    comp_pattern = re.compile(
        "[A-Z0-9]+\w*?{}\w*?".format(SUFFIX_MAP[suffix])
    )

    rxns = []

    for rxn in model.reactions:
        if comp_pattern.match(rxn.id):
            if ('biomass' not in rxn.id.lower()) and (
                    rxn not in transport_rxns and
                    rxn not in exchange_demand_rxns):
                rxns.append(rxn)

    return [rxn for rxn in rxns if suffix not in rxn.compartments]


def find_reaction_tag_transporter(model):
    """
    Return incorrectly tagged transport reactions.

    A transport reaction is defined as follows:
       -- It contains metabolites from at least 2 compartments
       -- At least 1 metabolite undergoes no chemical reaction
          i.e. the formula stays the same on both sides of the equation

    Reactions that only transport protons ('H') across the membrane are
    excluded, as well as reactions with redox cofactors whose formula is
    either 'X' or 'XH2'

    Parameters
    ----------
    model : cobra.Model
            A cobrapy metabolic model
    """
    transport_rxns = find_transport_reactions(model)
    atp_adp_rxns = find_atp_adp_converting_reactions(model)

    non_abc_transporters = set(transport_rxns).difference(set(atp_adp_rxns))

    return [rxn for rxn in non_abc_transporters
            if not re.match("[A-Z0-9]+\w*?t\w*?", rxn.id)]


def find_abc_tag_transporter(model):
    """
    Find Atp-binding cassette transport rxns without 'abc' tag.

    An ABC transport reaction is defined as follows:
       -- It contains metabolites from at least 2 compartments
       -- At least 1 metabolite undergoes no chemical reaction
          i.e. the formula stays the same on both sides of the equation
       -- ATP is converted to ADP (+ Pi + H,
          yet this isn't checked for explicitly)

    Reactions that only transport protons ('H') across the membrane are
    excluded, as well as reactions with redox cofactors whose formula is
    either 'X' or 'XH2'

    Parameters
    ----------
    model : cobra.Model
            A cobrapy metabolic model
    """
    transport_rxns = find_transport_reactions(model)
    atp_adp_rxns = find_atp_adp_converting_reactions(model)

    abc_transporters = set(transport_rxns).intersection(set(atp_adp_rxns))

    return [rxn for rxn in abc_transporters
            if not re.match("[A-Z0-9]+\w*?abc\w*?", rxn.id)]


def find_upper_case_mets(model):
    """
    Find metabolites whose ID roots contain uppercase characters.

    In metabolite names, individual uppercase exceptions are allowed:
    -- Dextorotary prefix: 'D'
    -- Levorotary prefix: 'L'
    -- Prefixes for the absolute configuration of a stereocenter: 'R' and 'S'
    -- Prefixes for the absolute stereochemistry of double bonds 'E' and 'Z'
    -- Acyl carrier proteins can be labeled as 'ACP'

    Parameters
    ----------
    model : cobra.Model
            A cobrapy metabolic model
    """
    comp_pattern = "^([a-z0-9]|Z|E|L|D|ACP|S|R)+_\w+"
    return [met for met in model.metabolites
            if not re.match(comp_pattern, met.id)]


def find_untagged_demand_rxns(model):
    """
    Find demand reactions whose IDs do not begin with 'DM_'.

    [1] defines demand reactions as:
    -- 'unbalanced network reactions that allow the accumulation of a compound'
    -- reactions that are chiefly added during the gap-filling process
    -- as a means of dealing with 'compounds that are known to be produced by
    the organism [..] (i) for which no information is available about their
    fractional distribution to the biomass or (ii) which may only be produced
    in some environmental conditions
    -- reactions with a formula such as: 'met_c -> '

    Demand reactions differ from exchange reactions in that the metabolites
    are not removed from the extracellular environment, but from any of the
    organism's compartments.

    Parameters
    ----------
    model : cobra.Model
            A cobrapy metabolic model

    References
    ----------
    [1] Thiele, I., & Palsson, B. Ø. (2010, January). A protocol for generating
    a high-quality genome-scale metabolic reconstruction. Nature protocols.
    Nature Publishing Group. http://doi.org/10.1038/nprot.2009.203
    """
    return [rxn for rxn in model.exchanges if ('e' not in rxn.compartments) and
            not rxn.id.startswith("DM_")]


def find_untagged_exchange_rxns(model):
    """
    Find exchange reactions whose IDs do not begin with 'EX_'.

    [1] defines exchange reactions as:
    -- reactions that 'define the extracellular environment'
    -- 'unbalanced, extra-organism reactions that represent the supply to or
    removal of metabolites from the extra-organism "space"'
    -- reactions with a formula such as: 'met_e -> ' or ' -> met_e' or
    'met_e <=> '

    Exchange reactions differ from demand reactions in that the metabolites
    are removed from or added to the extracellular environment only. With this
    the updake or secretion of a metabolite is modeled, respectively.

    Parameters
    ----------
    model : cobra.Model
            A cobrapy metabolic model

    References
    ----------
    [1] Thiele, I., & Palsson, B. Ø. (2010, January). A protocol for generating
    a high-quality genome-scale metabolic reconstruction. Nature protocols.
    Nature Publishing Group. http://doi.org/10.1038/nprot.2009.203
    """
    return [rxn for rxn in model.exchanges if ('e' in rxn.compartments) and
            not rxn.id.startswith("EX_")]
