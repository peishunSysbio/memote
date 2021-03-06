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

"""
Biomass tests performed on an instance of ``cobra.Model``.

N.B.: We parametrize each function here with the identified biomass reactions.
In the storage of test results we rely on the order of the biomass fixtures
to remain the same as the parametrized test cases.
"""

from __future__ import absolute_import

import logging

import pytest
import numpy as np

import memote.support.biomass as biomass
import memote.support.helpers as helpers
from memote.utils import annotate, truncate, get_ids, wrapper


LOGGER = logging.getLogger(__name__)
BIOMASS_IDS = pytest.memote.biomass_ids


@annotate(title="Presence of a Biomass Reaction", type="array")
def test_biomass_presence():
    """
    Expect the model to contain at least one biomass reaction.

    The biomass composition aka biomass formulation aka biomass reaction
    is a common pseudo-reaction accounting for biomass synthesis in
    constraints-based modelling. It describes the stoichiometry of
    intracellular compounds that are required for cell growth.
    """
    ann = test_biomass_presence.annotation
    ann["data"] = BIOMASS_IDS
    ann["message"] = wrapper.fill(
        """In this model {} the following biomass reactions were
        identified: {}""".format(
            len(ann["data"]), truncate(ann["data"])))
    assert len(ann["data"]) > 0, ann["message"]


@pytest.mark.parametrize("reaction_id", BIOMASS_IDS)
@annotate(title="Biomass Consistency", type="object", data=dict(),
          message=dict())
def test_biomass_consistency(read_only_model, reaction_id):
    """Expect biomass components to sum up to 1 g[CDW]."""
    ann = test_biomass_consistency.annotation
    reaction = read_only_model.reactions.get_by_id(reaction_id)
    ann["data"][reaction_id] = biomass.sum_biomass_weight(reaction)
    ann["message"][reaction_id] = wrapper.fill(
        """The component molar mass of the biomass reaction {} sums up to {}
        which is outside of the 1e-03 margin from 1 mmol / g[CDW] / h.
        """.format(reaction_id, ann["data"][reaction_id]))
    assert np.isclose(
        ann["data"][reaction_id], 1.0, atol=1e-03), ann["message"][reaction_id]


@pytest.mark.parametrize("reaction_id", BIOMASS_IDS)
@annotate(title="Biomass Production At Default State", type="object",
          data=dict(), message=dict())
def test_biomass_default_production(model, reaction_id):
    """Expect biomass production in default medium."""
    ann = test_biomass_default_production.annotation
    ann["data"][reaction_id] = helpers.run_fba(model, reaction_id)
    ann["message"][reaction_id] = wrapper.fill(
        """Using the biomass reaction {} this is the growth rate that can be
        achieved when the model is simulated on the provided default medium: {}
        """.format(reaction_id, ann["data"][reaction_id]))
    assert ann["data"][reaction_id] > 0.0, ann["message"][reaction_id]


@pytest.mark.parametrize("reaction_id", BIOMASS_IDS)
@annotate(title="Blocked Biomass Precursors At Default State", type="object",
          data=dict(), message=dict())
def test_biomass_precursors_default_production(read_only_model, reaction_id):
    """Expect production of all biomass precursors in default medium."""
    ann = test_biomass_precursors_default_production.annotation
    reaction = read_only_model.reactions.get_by_id(reaction_id)
    ann["data"][reaction_id] = get_ids(
        biomass.find_blocked_biomass_precursors(reaction, read_only_model)
    )
    ann["message"][reaction_id] = wrapper.fill(
        """Using the biomass reaction {} and when the model is simulated on the
        provided default medium a total of {} precursors cannot be produced: {}
        """.format(reaction_id, len(ann["data"][reaction_id]),
                   ann["data"][reaction_id]))
    assert len(ann["data"][reaction_id]) == 0, ann["message"][reaction_id]


@pytest.mark.parametrize("reaction_id", BIOMASS_IDS)
@annotate(title="Blocked Biomass Precursors In Complete Medium", type="object",
          data=dict(), message=dict())
def test_biomass_precursors_open_production(model, reaction_id):
    """Expect precursor production in complete medium."""
    ann = test_biomass_precursors_open_production.annotation
    with model:
        for exchange in model.exchanges:
            exchange.bounds = (-1000, 1000)
        reaction = model.reactions.get_by_id(reaction_id)
        ann["data"][reaction_id] = get_ids(
            biomass.find_blocked_biomass_precursors(reaction, model)
        )
    ann["message"][reaction_id] = wrapper.fill(
        """Using the biomass reaction {} and when the model is simulated in
        complete medium a total of {} precursors cannot be produced: {}
        """.format(reaction_id, len(ann["data"][reaction_id]),
                   ann["data"][reaction_id]))
    assert len(ann["data"][reaction_id]) == 0, ann["message"][reaction_id]


@pytest.mark.parametrize("reaction_id", BIOMASS_IDS)
@annotate(title="Growth-associated Maintenance in Biomass Reaction",
          type="object", data=dict(), message=dict())
def test_gam_in_biomass(model, reaction_id):
    """Expect the biomass reactions to contain atp and adp."""
    ann = test_gam_in_biomass.annotation
    reaction = model.reactions.get_by_id(reaction_id)
    ann["data"][reaction_id] = biomass.gam_in_biomass(reaction)
    ann["message"][reaction_id] = wrapper.fill(
        """{} does not contain a term for growth-associated maintenance.
        """.format(reaction_id))
    assert ann["data"][reaction_id], ann["message"][reaction_id]


@pytest.mark.parametrize("reaction_id", BIOMASS_IDS)
@annotate(title="Unrealistic Growth Rate In Default Condition", type='object',
          data=dict(), message=dict())
def test_fast_growth_default(model, reaction_id):
    """Expect the predicted growth rate for each BOF to be below 10.3972.

    This is based on lowest doubling time reported here
    http://www.pnnl.gov/science/highlights/highlight.asp?id=879
    """
    ann = test_fast_growth_default.annotation
    ann["data"][reaction_id] = helpers.run_fba(model, reaction_id)
    ann["message"][reaction_id] = wrapper.fill(
        """Using the biomass reaction {} and when the model is simulated on
        the provided default medium the growth rate amounts to {}""".format(
            reaction_id, ann["data"][reaction_id]))
    assert ann["data"][reaction_id] <= 10.3972, ann["message"][reaction_id]
