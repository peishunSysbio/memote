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

"""Perform basic tests on an instance of ``cobra.Model``."""

from __future__ import absolute_import, division

import memote.support.basic as basic
import memote.support.helpers as helpers
from memote.utils import annotate, get_ids, truncate, wrapper


@annotate(title="Model Identifier", type="string")
def test_model_id_presence(read_only_model):
    """Expect that the model has an identifier."""
    ann = test_model_id_presence.annotation
    assert hasattr(read_only_model, "id")
    ann["data"] = read_only_model.id
    assert bool(read_only_model.id)


@annotate(title="Total Number of Genes", type="length")
def test_genes_presence(read_only_model):
    """Expect that >= 1 genes are defined in the model."""
    ann = test_genes_presence.annotation
    assert hasattr(read_only_model, "genes")
    ann["data"] = get_ids(read_only_model.genes)
    ann["message"] = "{:d} genes are defined in the model.".format(
        len(ann["data"]))
    assert len(ann["data"]) >= 1, ann["message"]


@annotate(title="Total Number of Reactions", type="length")
def test_reactions_presence(read_only_model):
    """Expect that >= 1 reactions are present in the model."""
    ann = test_reactions_presence.annotation
    assert hasattr(read_only_model, "reactions")
    ann["data"] = get_ids(read_only_model.reactions)
    ann["message"] = "{:d} reactions are defined in the model.".format(
        len(ann["data"]))
    assert len(ann["data"]) >= 1, ann["message"]


@annotate(title="Total Number of Metabolites", type="length")
def test_metabolites_presence(read_only_model):
    """Expect that >= 1 metabolites are present in the model."""
    ann = test_metabolites_presence.annotation
    assert hasattr(read_only_model, "metabolites")
    ann["data"] = get_ids(read_only_model.metabolites)
    ann["message"] = "{:d} metabolites are defined in the model.".format(
        len(ann["data"]))
    assert len(ann["data"]) >= 1, ann["message"]


@annotate(title="Total Number of Transport Reactions", type="length")
def test_transport_reaction_presence(read_only_model):
    """Expect >= 1 transport reactions are present in the model."""
    ann = test_transport_reaction_presence.annotation
    ann["data"] = get_ids(helpers.find_transport_reactions(read_only_model))
    ann["message"] = wrapper.fill(
        """{:d} transport reactions are defined in the model.""".format(
            len(ann["data"])))
    assert len(ann["data"]) >= 1, ann["message"]


@annotate(title="Metabolites without Formula", type="length")
def test_metabolites_formula_presence(read_only_model):
    """
    Expect all metabolites to have a formula.

    To ensure that reactions are mass-balanced, all model metabolites
    ought to be provided with a formula.
    """
    ann = test_metabolites_formula_presence.annotation
    ann["data"] = get_ids(
        basic.check_metabolites_formula_presence(read_only_model))
    ann["metric"] = len(ann["data"]) / len(read_only_model.metabolites)
    ann["message"] = wrapper.fill(
        """There are a total of {}
        metabolites ({:.2%}) without a formula: {}""".format(
            len(ann["data"]), ann["metric"], truncate(ann["data"])))
    assert len(ann["data"]) == 0, ann["message"]


@annotate(title="Metabolites without Charge", type="length")
def test_metabolites_charge_presence(read_only_model):
    """
    Expect all metabolites to have charge information.

    To ensure that reactions are charge-balanced, all model metabolites
        ought to be provided with a charge.
    """
    ann = test_metabolites_charge_presence.annotation
    ann["data"] = get_ids(
        basic.check_metabolites_charge_presence(read_only_model))
    ann["metric"] = len(ann["data"]) / len(read_only_model.metabolites)
    ann["message"] = wrapper.fill(
        """There are a total of {}
        metabolites ({:.2%}) without a charge: {}""".format(
            len(ann["data"]), ann["metric"], truncate(ann["data"])))
    assert len(ann["data"]) == 0, ann["message"]


@annotate(title="Reactions without GPR", type="length")
def test_gene_protein_reaction_rule_presence(read_only_model):
    """
    Expect all non-exchange reactions to have a GPR rule.

    Gene-Protein-Reaction rules express which gene has what function.
    The presence of this annotation is important to justify the existence
    of reactions in the model, and is required to conduct in silico gene
    deletion studies. However, reactions without GPR may also be valid:
    Spontaneous reactions, or known reactions with yet undiscovered genes
    likely lack GPR.
    """
    ann = test_gene_protein_reaction_rule_presence.annotation
    missing_gpr_metabolic_rxns = set(
        basic.check_gene_protein_reaction_rule_presence(read_only_model)
    ).difference(set(read_only_model.exchanges))
    ann["data"] = get_ids(missing_gpr_metabolic_rxns)
    ann["metric"] = len(ann["data"]) / len(read_only_model.reactions)
    ann["message"] = wrapper.fill(
        """There are a total of {} reactions ({:.2%}) without GPR:
        {}""".format(len(ann["data"]), ann["metric"], truncate(ann["data"])))
    assert len(ann["data"]) == 0, ann["message"]


@annotate(title="Non-Growth Associated Maintenance Reaction", type="length")
def test_ngam_presence(read_only_model):
    """
    Expect a single non growth-associated maintenance reaction.

    The Non-Growth Associated Maintenance reaction (NGAM) is an
    ATP-hydrolysis reaction added to metabolic models to represent energy
    expenses that the cell invests in continuous processes independent of
    the growth rate.
    """
    ann = test_ngam_presence.annotation
    ann["data"] = get_ids(basic.find_ngam(read_only_model))
    ann["message"] = wrapper.fill(
        """A total of {} NGAM reactions could be identified:
        {}""".format(len(ann["data"]), truncate(ann["data"])))
    assert len(ann["data"]) == 1, ann["message"]


@annotate(title="Metabolic Coverage", type="number")
def test_metabolic_coverage(read_only_model):
    """
    Expect a model to have a metabolic coverage >= 1.

    The degree of metabolic coverage indicates the modeling detail of a
    given reconstruction calculated by dividing the total amount of
    reactions by the amount of genes. Models with a 'high level of modeling
    detail have ratios >1, and models with a low level of detail have
    ratios <1. This difference arises as models with basic or intermediate
    levels of detail are assumed to include many reactions in which several
    gene products and their enzymatic transformations are ‘lumped’.
    """
    ann = test_metabolic_coverage.annotation
    ann["metric"] = basic.calculate_metabolic_coverage(read_only_model)
    ann["message"] = wrapper.fill(
        """The degree of metabolic coverage is {:.2}.""".format(ann["metric"]))
    assert ann["metric"] >= 1, ann["message"]


@annotate(title="Total Number of Compartments", type="length")
def test_compartments_presence(read_only_model):
    """Expect that >= 3 compartments are defined in the model."""
    ann = test_compartments_presence.annotation
    assert hasattr(read_only_model, "compartments")
    ann["data"] = list(read_only_model.get_metabolite_compartments())
    ann["message"] = wrapper.fill(
        """A total of {:d} compartments are defined in the model: {}""".format(
            len(ann["data"]), truncate(ann["data"])))
    assert len(ann["data"]) >= 3, ann["message"]


@annotate(title="Number of Enzyme Complexes", type="length")
def test_enzyme_complex_presence(read_only_model):
    """Expect that >= 1 enzyme complexes are present in the model."""
    ann = test_enzyme_complex_presence.annotation
    ann["data"] = list(basic.find_enzyme_complexes(read_only_model))
    ann["message"] = wrapper.fill(
        """A total of {:d} enzyme complexes are defined through GPR rules in
        the model.""".format(len(ann["data"])))
    assert len(ann["data"]) >= 1, ann["message"]


@annotate(title="Number of Purely Metabolic Reactions", type="length")
def test_find_pure_metabolic_reactions(read_only_model):
    """Expect >= 1 pure metabolic reactions are present in the model."""
    ann = test_find_pure_metabolic_reactions.annotation
    ann["data"] = get_ids(
        basic.find_pure_metabolic_reactions(read_only_model))
    ann["metric"] = len(ann["data"]) / len(read_only_model.reactions)
    ann["message"] = wrapper.fill(
        """A total of {:d} ({:.2%}) purely metabolic reactions are defined in
        the model, this excludes transporters, exchanges, or pseudo-reactions:
        {}""".format(len(ann["data"]), ann["metric"], truncate(ann["data"])))
    assert len(ann["data"]) >= 1, ann["message"]


@annotate(title="Number of Transport Reactions", type="length")
def test_find_transport_reactions(read_only_model):
    """Expect >= 1 transport reactions are present in the read_only_model."""
    ann = test_find_transport_reactions.annotation
    ann["data"] = get_ids(helpers.find_transport_reactions(read_only_model))
    ann["metric"] = len(ann["data"]) / len(read_only_model.reactions)
    ann["message"] = wrapper.fill(
        """A total of {:d} ({:.2%}) transport reactions are defined in the
        model, this excludes purely metabolic reactions, exchanges, or
        pseudo-reactions: {}""".format(
            len(ann["data"]), ann["metric"], truncate(ann["data"])))
    assert len(ann["data"]) >= 1, ann["message"]


@annotate(title="Number of Unique Metabolites", type="length")
def test_find_unique_metabolites(read_only_model):
    """Expect there to be less metabolites when removing compartment tag."""
    ann = test_find_unique_metabolites.annotation
    ann["data"] = list(basic.find_unique_metabolites(read_only_model))
    ann["metric"] = len(ann["data"]) / len(read_only_model.metabolites)
    ann["message"] = wrapper.fill(
        """Not counting the same entities in other compartments, there is a
        total of {} ({:.2%}) unique metabolites in the model: {}""".format(
            len(ann["data"]), ann["metric"], truncate(ann["data"])))
    assert len(ann["data"]) < len(read_only_model.metabolites), ann["message"]
