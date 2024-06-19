#
# Copyright (C) 2024 by frePPLe bv
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

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("forecast", "0006_default_parameters"),
        ("execute", "0010_dataexport"),
    ]

    operations = [
        migrations.RunSQL(
            """
            insert into execute_export
               (name, sql, report, arguments)
               values (
                 'forecastreport.csv.gz',
                 null,
                 'freppledb.forecast.views.OverviewReport',
                 '{"format": "csvlist", "buckets": "month", "horizontype": true, "horizonunit": "month", "horizonlength": 6}'
               )
            on conflict (name) do nothing
            """,
            """
            delete from execute_export where name = 'forecastreport.csv.gz'
            """,
        ),
    ]
