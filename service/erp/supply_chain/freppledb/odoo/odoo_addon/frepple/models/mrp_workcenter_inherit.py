# -*- coding: utf-8 -*-
#
# Copyright (C) 2014 by frePPLe bv
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
from odoo import models, fields


class WorkcenterInherit(models.Model):
    _inherit = "mrp.workcenter"

    owner = fields.Many2one(
        "mrp.workcenter",
        "Owner",
        required=False,
        help="Groups workcenters together in groups",
    )
    tool = fields.Boolean(
        "is a tool",
        default=False,
        help="Mark workcenters that are tools, fixtures or holders. The same tool needs to accompany a manufacturing order through all its work orders.",
    )
