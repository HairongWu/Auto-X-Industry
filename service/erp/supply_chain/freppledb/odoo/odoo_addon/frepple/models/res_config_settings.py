# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 by frePPLe bv
#
# This library is free software; you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero
# General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public
# License along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

from odoo import fields, models
from .. import with_mrp


class ResConfigSettings(models.TransientModel):
    _inherit = "res.config.settings"

    if with_mrp:
        manufacturing_warehouse = fields.Many2one(
            "stock.warehouse",
            "Manufacturing warehouse",
            related="company_id.manufacturing_warehouse",
            readonly=False,
        )
        calendar = fields.Many2one(
            "resource.calendar",
            "Calendar",
            related="company_id.calendar",
            readonly=False,
        )
    webtoken_key = fields.Char(
        "Webtoken key", size=128, related="company_id.webtoken_key", readonly=False
    )
    frepple_server = fields.Char(
        "frePPLe server", size=128, related="company_id.frepple_server", readonly=False
    )
    respect_reservations = fields.Boolean(
        related="company_id.respect_reservations", readonly=False
    )
    disclose_stack_trace = fields.Boolean(
        related="company_id.disclose_stack_trace",
        readonly=False,
    )

