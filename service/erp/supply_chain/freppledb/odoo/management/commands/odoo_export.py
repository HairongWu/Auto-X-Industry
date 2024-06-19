#
# Copyright (C) 2016 by frePPLe bv
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

import base64
from datetime import datetime, timedelta
import email
import itertools
import json
import jwt
import math
import time
from xml.sax.saxutils import quoteattr
from urllib.request import urlopen, Request, HTTPError, URLError

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from django.db import DEFAULT_DB_ALIAS
from django.utils.translation import gettext_lazy as _
from django.template import Template, RequestContext

from freppledb import __version__
from freppledb.common.middleware import _thread_locals
from freppledb.common.models import User, Parameter
from freppledb.execute.models import Task
from freppledb.input.models import (
    PurchaseOrder,
    DistributionOrder,
    ManufacturingOrder,
    Demand,
)


class Command(BaseCommand):
    help = "Exports frepple data to odoo."

    requires_system_checks = []

    recordsperpage = 100

    def get_version(self):
        return __version__

    def add_arguments(self, parser):
        parser.add_argument("--user", help="User running the command")
        parser.add_argument(
            "--database",
            default=DEFAULT_DB_ALIAS,
            help="Nominates the frePPLe database to export",
        )
        parser.add_argument(
            "--task",
            type=int,
            help="Task identifier (generated automatically if not provided)",
        )

    def handle(self, **options):
        self.verbosity = int(options["verbosity"])
        self.database = options["database"]
        if self.database not in settings.DATABASES.keys():
            raise CommandError("No database settings known for '%s'" % self.database)

        if options["user"]:
            try:
                user = (
                    User.objects.all()
                    .using(self.database)
                    .get(username=options["user"])
                )
            except Exception:
                raise CommandError("User '%s' not found" % options["user"])
        else:
            user = None

        now = datetime.now()
        task = None
        old_thread_locals = getattr(_thread_locals, "database", None)
        try:
            # Initialize the task
            setattr(_thread_locals, "database", self.database)
            if "task" in options and options["task"]:
                try:
                    task = (
                        Task.objects.all().using(self.database).get(pk=options["task"])
                    )
                except Exception:
                    raise CommandError("Task identifier not found")
                if (
                    task.started
                    or task.finished
                    or task.status != "Waiting"
                    or task.name != "odoo_export"
                ):
                    raise CommandError("Invalid task identifier")
                task.status = "0%"
                task.started = now
            else:
                task = Task(
                    name="odoo_export",
                    submitted=now,
                    started=now,
                    status="0%",
                    user=user,
                )
            task.message = "Exporting plan data to odoo"
            task.save(using=self.database)

            # Get parameters
            missing = []
            self.odoo_user = Parameter.getValue("odoo.user", self.database)
            if not self.odoo_user:
                missing.append("odoo_user")
            self.odoo_password = settings.ODOO_PASSWORDS.get(self.database, None)
            if not self.odoo_password:
                self.odoo_password = Parameter.getValue("odoo.password", self.database)
            if not self.odoo_password:
                missing.append("odoo_password")
            self.odoo_db = Parameter.getValue("odoo.db", self.database)
            if not self.odoo_db:
                missing.append("odoo_db")
            self.odoo_url = Parameter.getValue("odoo.url", self.database)
            if not self.odoo_url:
                missing.append("odoo_url")
            if not self.odoo_url.endswith("/"):
                self.odoo_url += "/"
            self.odoo_company = Parameter.getValue("odoo.company", self.database)
            if not self.odoo_company:
                missing.append("odoo_company")
            self.odoo_language = Parameter.getValue(
                "odoo.language", self.database, "en_US"
            )
            if not self.odoo_language:
                missing.append("odoo_language")
            if missing:
                raise CommandError("Missing parameter %s" % ", ".join(missing))

            # First loop with only purpose to count the records...
            self.exported = []
            for i in self.generateOperationPlansToPublish():
                pass
            total_pages = math.ceil(len(self.exported) / self.recordsperpage)
            self.demand_count = 0
            for i in self.generateDemandsToPublish():
                self.demand_count += 1
            total_pages += math.ceil(self.demand_count / self.recordsperpage)

            # Collect data to send
            counter = 1
            self.exported = []
            self.boundary = email.generator._make_boundary()
            # Track if this is the first page we send
            # for cleaning POs/MOs in Odoo
            self.firstPage = True
            for page in self.generatePagesToPublish(
                records_per_page=self.recordsperpage
            ):
                # Connect to the odoo URL to POST data
                encoded = base64.encodebytes(
                    ("%s:%s" % (self.odoo_user, self.odoo_password)).encode("utf-8")
                )
                size = len(page)
                req = Request(
                    "%sfrepple/xml/" % self.odoo_url,
                    data=page,
                    headers={
                        "Authorization": "Basic %s" % encoded.decode("ascii")[:-1],
                        "Content-Type": "multipart/form-data; boundary=%s"
                        % self.boundary,
                        "Content-length": size,
                    },
                )
                with urlopen(req) as f:
                    # Read the odoo response
                    f.read()

                # Mark the exported operations as approved
                for i in self.exported:
                    i.status = "approved"
                    i.source = "odoo_1"
                    i.save(using=self.database, update_fields=("status", "source"))

                # Progress
                task.status = "%s%%" % math.ceil(counter / total_pages * 100)
                task.message = "Sent page %s of %s with plan data to odoo" % (
                    counter,
                    total_pages,
                )
                task.save(using=self.database, update_fields=("status", "message"))
                counter += 1
                self.exported = []

            # Task update
            del self.exported
            task.status = "Done"
            task.message = None
            task.finished = datetime.now()
            task.processid = None

        except HTTPError as e:
            if task:
                task.status = "Failed"
                task.message = "Internal server error %s on odoo side" % e.code
                task.finished = datetime.now()
                task.processid = None
            raise e

        except URLError as e:
            if task:
                task.status = "Failed"
                task.message = "Couldn't connect to odoo"
                task.finished = datetime.now()
                task.processid = None
            raise e

        except Exception as e:
            if task:
                task.status = "Failed"
                task.message = "%s" % e
                task.finished = datetime.now()
                task.processid = None
            raise e

        finally:
            if task:
                task.save(using=self.database)
            setattr(_thread_locals, "database", old_thread_locals)

    def generatePagesToPublish(self, records_per_page=None):
        if not records_per_page:
            records_per_page = self.recordsperpage
        cnt = 0
        output = []
        for rec in self.generateOperationPlansToPublish():
            output.append(rec)
            cnt += 1
            if cnt >= records_per_page:
                yield self.buildPage(output, "operationplans")
                output = []
                cnt = 0
        if cnt:
            yield self.buildPage(output, "operationplans")
        cnt = 0
        output = []
        for rec in self.generateDemandsToPublish():
            output.append(rec)
            cnt += 1
            if cnt >= records_per_page:
                yield self.buildPage(output, "demands")
                output = []
                cnt = 0
        if cnt:
            yield self.buildPage(output, "demands")

    def buildPage(self, output, objtype):
        token = jwt.encode(
            {"exp": round(time.time()) + 600, "user": self.odoo_user},
            settings.DATABASES[self.database].get(
                "SECRET_WEBTOKEN_KEY", settings.SECRET_KEY
            ),
            algorithm="HS256",
        )
        if not isinstance(token, str):
            token = token.decode("ascii")
        header = [
            "--%s\r" % self.boundary,
            'Content-Disposition: form-data; name="webtoken"\r',
            "\r",
            "%s\r" % token,
            "--%s\r" % self.boundary,
            'Content-Disposition: form-data; name="database"\r',
            "\r",
            "%s\r" % self.odoo_db,
            "--%s\r" % self.boundary,
            'Content-Disposition: form-data; name="language"\r',
            "\r",
            "%s\r" % self.odoo_language,
            "--%s\r" % self.boundary,
            'Content-Disposition: form-data; name="company"\r',
            "\r",
            "%s\r" % self.odoo_company,
            "--%s\r" % self.boundary,
            'Content-Disposition: form-data; name="mode"\r',
            "\r",
            "%s\r"
            % (
                1 if self.firstPage else 3
            ),  # 3 is for incremental export with a skip of quantity aggregation in inbound.py
            "--%s\r" % self.boundary,
            'Content-Disposition: file; name="frePPLe plan"; filename="frepple_plan.xml"\r',
            "Content-Type: application/xml\r",
            "\r",
            '<?xml version="1.0" encoding="UTF-8" ?>',
            '<plan xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">',
            "<%s>" % objtype,
        ]
        footer = [
            "</%s>" % objtype,
            "</plan>",
            "--%s--\r" % self.boundary,
            "\r",
        ]
        self.firstPage = False
        return "\n".join(itertools.chain(header, output, footer)).encode("utf-8")

    def generateOperationPlansToPublish(self):
        today = datetime.today()
        # Purchase orders to export
        for i in (
            PurchaseOrder.objects.using(self.database)
            .filter(
                status="proposed",
                item__source__startswith="odoo",
                startdate__lte=today + timedelta(days=7),
            )
            .exclude(item__type="make to order")
            .order_by("startdate")
            .select_related("location", "item", "supplier")
        ):
            if (
                not i.item
                or not i.item.source
                or not i.item.subcategory
                or not i.location.subcategory
                or not i.item.source.startswith("odoo")
                or i.supplier.name == "Unknown supplier"
            ):
                continue
            self.exported.append(i)
            yield '<operationplan reference=%s ordertype="PO" item=%s location=%s supplier=%s start="%s" end="%s" quantity="%s" location_id=%s item_id=%s criticality="%d" batch=%s/>' % (
                quoteattr(i.reference),
                quoteattr(i.item.name),
                quoteattr(i.location.name),
                quoteattr(i.supplier.name),
                i.startdate,
                i.enddate,
                i.quantity,
                quoteattr(i.location.subcategory),
                quoteattr(i.item.subcategory),
                int(i.criticality),
                quoteattr(i.batch or ""),
            )

        # Distribution orders to export
        for i in (
            DistributionOrder.objects.using(self.database)
            .filter(
                status="proposed",
                item__source__startswith="odoo",
                startdate__lte=today + timedelta(days=7),
            )
            .exclude(item__type="make to order")
            .order_by("startdate")
            .select_related("item", "origin", "destination")
        ):
            if (
                not i.item
                or not i.item.source
                or not i.item.subcategory
                or not i.origin.subcategory
                or not i.destination.subcategory
                or not i.item.source.startswith("odoo")
            ):
                continue
            self.exported.append(i)
            yield '<operationplan status="%s" reference=%s ordertype="DO" item=%s origin=%s destination=%s start="%s" end="%s" quantity="%s" origin_id=%s destination_id=%s item_id=%s criticality="%d" batch=%s/>' % (
                i.status,
                quoteattr(i.reference),
                quoteattr(i.item.name),
                quoteattr(i.origin.name),
                quoteattr(i.destination.name),
                i.startdate,
                i.enddate,
                i.quantity,
                quoteattr(i.origin.subcategory),
                quoteattr(i.destination.subcategory),
                quoteattr(i.item.subcategory),
                int(i.criticality),
                quoteattr(i.batch or ""),
            )

        # Manufacturing orders to export
        for i in (
            ManufacturingOrder.objects.using(self.database)
            .filter(
                item__source__startswith="odoo",
                operation__source__startswith="odoo",
                operation__owner__isnull=True,
                status="proposed",
                startdate__lte=today + timedelta(days=7),
            )
            .exclude(item__type="make to order")
            .order_by("startdate")
            .select_related("operation", "location", "item", "owner")
        ):
            if (
                not i.operation
                or not i.operation.source
                or not i.operation.item
                or not i.operation.source.startswith("odoo")
                or not i.item.subcategory
                or not i.location.subcategory
            ):
                continue
            self.exported.append(i)
            demand_str = json.dumps(i.plan["pegging"]) if i.plan["pegging"] else ""
            if i.operation.category == "subcontractor":
                yield '<operationplan ordertype="PO" id=%s item=%s location=%s supplier=%s start="%s" end="%s" quantity="%s" location_id=%s item_id=%s criticality="%d" batch=%s/>' % (
                    quoteattr(i.reference),
                    quoteattr(i.item.name),
                    quoteattr(i.location.name),
                    quoteattr(i.operation.subcategory or ""),
                    i.startdate,
                    i.enddate,
                    i.quantity,
                    quoteattr(i.location.subcategory or ""),
                    quoteattr(i.item.subcategory or ""),
                    int(i.criticality),
                    quoteattr(i.batch or ""),
                )
            else:
                rec = [
                    '<operationplan reference=%s ordertype="MO" item=%s location=%s operation=%s start="%s" end="%s" quantity="%s" location_id=%s item_id=%s criticality="%d" demand=%s batch=%s>'
                    % (
                        quoteattr(i.reference),
                        quoteattr(i.operation.item.name),
                        quoteattr(i.operation.location.name),
                        quoteattr(i.operation.name),
                        i.startdate,
                        i.enddate,
                        i.quantity,
                        quoteattr(i.operation.location.subcategory),
                        quoteattr(i.operation.item.subcategory),
                        int(i.criticality),
                        quoteattr(demand_str),
                        quoteattr(i.batch or ""),
                    )
                ]
                wolist = [i for i in i.xchildren.using(self.database).all()]
                if wolist:
                    for wo in wolist:
                        rec.append(
                            '<workorder operation=%s start="%s" end="%s">'
                            % (
                                quoteattr(wo.operation.name),
                                wo.startdate,
                                wo.enddate,
                            )
                        )
                        for wores in wo.resources.using(self.database).all():
                            if (
                                wores.resource.source
                                and wores.resource.source.startswith("odoo")
                            ):
                                rec.append(
                                    "<resource name=%s id=%s/>"
                                    % (
                                        quoteattr(wores.resource.name),
                                        quoteattr(wores.resource.category or ""),
                                    )
                                )
                        rec.append("</workorder>")
                else:
                    for opplanres in i.resources.using(self.database).all():
                        if (
                            opplanres.resource.source
                            and opplanres.resource.source.startswith("odoo")
                        ):
                            rec.append(
                                "<resource name=%s id=%s/>"
                                % (
                                    quoteattr(opplanres.resource.name),
                                    quoteattr(opplanres.resource.category or ""),
                                )
                            )
                rec.append("</operationplan>")
                yield "".join(rec)

        # Work orders to export
        # We don't create work orders, but only updates existing work orders.
        # We leave it to odoo to create the workorders for a new manufacturing order.
        for i in (
            ManufacturingOrder.objects.using(self.database)
            .filter(
                owner__operation__type="routing",
                operation__source__startswith="odoo",
                owner__item__source__startswith="odoo",
                status="approved",
            )
            .order_by("startdate")
            .select_related("operation", "location", "item", "owner")
        ):
            if (
                i.status not in ("proposed", "approved")
                or not i.operation
                or not i.operation.source
                or not i.owner.operation.item
                or not i.operation.source.startswith("odoo")
                or not i.owner.item.subcategory
                or not i.location.subcategory
            ):
                continue
            self.exported.append(i)
            rec = [
                '<operationplan reference=%s owner=%s ordertype="WO" item=%s location=%s operation=%s start="%s" end="%s" quantity="%s" location_id=%s item_id=%s batch=%s>'
                % (
                    quoteattr(i.reference),
                    quoteattr(i.owner.reference),
                    quoteattr(i.owner.operation.item.name),
                    quoteattr(i.operation.location.name),
                    quoteattr(i.operation.name),
                    i.startdate,
                    i.enddate,
                    i.quantity,
                    quoteattr(i.operation.location.subcategory),
                    quoteattr(i.owner.operation.item.subcategory),
                    quoteattr(i.batch or ""),
                )
            ]
            for wores in i.resources.using(self.database).all():
                if wores.resource.source and wores.resource.source.startswith("odoo"):
                    rec.append(
                        "<resource name=%s id=%s/>"
                        % (
                            quoteattr(wores.resource.name),
                            quoteattr(wores.resource.category or ""),
                        )
                    )
            rec.append("</operationplan>")
            yield "".join(rec)

    def generateDemandsToPublish(self):
        """
        Collect the delivery date of all open sales orders. This can used to populate
        the planned delivery date as information in odoo.
        """
        for d in (
            Demand.objects.using(self.database)
            .filter(
                source__startswith="odoo", deliverydate__isnull=False, status="open"
            )
            .only("name", "deliverydate")
        ):
            yield (
                '<demand ordertype="SO" name=%s deliverydate="%s"/>'
                % (
                    quoteattr(d.name),
                    d.deliverydate,
                )
            )

    # accordion template
    title = _("Export data to %(erp)s") % {"erp": "odoo"}
    index = 1450
    help_url = "command-reference.html#odoo_export"

    @staticmethod
    def getHTML(request):
        return Template(
            """
            {% load i18n %}
            <form role="form" method="post" action="{{request.prefix}}/execute/launch/odoo_export/">{% csrf_token %}
            <table>
              <tr>
                <td style="vertical-align:top; padding: 15px">
                   <button  class="btn btn-primary"  type="submit" value="{% trans "launch"|capfirst %}">{% trans "launch"|capfirst %}</button>
                </td>
                <td  style="padding: 0px 15px;">{% trans "Export frePPLe data to odoo." %}
                </td>
              </tr>
            </table>
            </form>
          """
        ).render(RequestContext(request))
