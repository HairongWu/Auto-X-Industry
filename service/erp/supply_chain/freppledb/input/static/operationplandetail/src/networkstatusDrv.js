/*
 * Copyright (C) 2017 by frePPLe bv
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE
 *
 */

'use strict';

angular.module('operationplandetailapp').directive('shownetworkstatusDrv', shownetworkstatusDrv);

shownetworkstatusDrv.$inject = ['$window', 'gettextCatalog'];

function shownetworkstatusDrv($window, gettextCatalog) {

  var directive = {
    restrict: 'EA',
    scope: { operationplan: '=data' },
    link: linkfunc
  };
  return directive;

  function linkfunc(scope, elem, attrs) {
    var template = '<div class="card-header"><h5 class="card-title" style="text-transform: capitalize">' +
      gettextCatalog.getString("network status") +
      '</h5></div><div class="card-body">' +
      '<table class="table table-sm table-hover table-borderless"><thead><tr><td>' +
      '<b style="text-transform: capitalize;">' + gettextCatalog.getString("item") + '</b>' +
      '</td><td>' +
      '<b style="text-transform: capitalize;">' + gettextCatalog.getString("location") + '</b>' +
      '</td><td>' +
      '<b style="text-transform: capitalize;">' + gettextCatalog.getString("onhand") + '</b>' +
      '</td><td>' +
      '<b style="text-transform: capitalize;">' + gettextCatalog.getString("purchase orders") + '</b>' +
      '</td><td>' +
      '<b style="text-transform: capitalize;">' + gettextCatalog.getString("distribution orders") + '</b>' +
      '</td><td>' +
      '<b style="text-transform: capitalize;">' + gettextCatalog.getString("manufacturing orders") + '</b>' +
      '</td><td>' +
      '<b style="text-transform: capitalize;">' + gettextCatalog.getString("overdue sales orders") + '</b>' +
      '</td><td>' +
      '<b style="text-transform: capitalize;">' + gettextCatalog.getString("sales orders") + '</b>' +
      '</td></tr></thead>' +
      '<tbody></tbody>' +
      '</table></div>';

    scope.$watchGroup(['operationplan.id', 'operationplan.network.length'], function (newValue, oldValue) {
      angular.element(document).find('#attributes-networkstatus').empty().append(template);
      var rows = '<tr><td colspan="8">' + gettextCatalog.getString('no network information') + '</td></tr>';

      if (typeof scope.operationplan !== 'undefined') {
        if (scope.operationplan.hasOwnProperty('network')) {
          rows = '';
          angular.forEach(scope.operationplan.network, function (thenetwork) {
            rows += '<tr><td>' + $.jgrid.htmlEncode(thenetwork[0])
              + "<a href=\"" + url_prefix + "/detail/input/item/" + admin_escape(thenetwork[0])
              + "/\" onclick='event.stopPropagation()'><span class='ps-2 fa fa-caret-right'></span></a>";
            if (thenetwork[1] === true) {
              rows += '<small>' + gettextCatalog.getString('superseded') + '</small>';
            }
            rows += '</td><td>'
              + $.jgrid.htmlEncode(thenetwork[2])
              + "<a href=\"" + url_prefix + "/detail/input/location/" + admin_escape(thenetwork[2])
              + "/\" onclick='event.stopPropagation()'><span class='ps-2 fa fa-caret-right'></span></a>"
              + '</td><td>'
              + grid.formatNumber(thenetwork[3])
              + '</td><td>'
              + grid.formatNumber(thenetwork[4]);

            if (thenetwork[4] > 0) {
              rows += "<a href=\"" + url_prefix + "/data/input/operationplanmaterial/buffer/"
                + admin_escape(thenetwork[0] + " @ " + thenetwork[2])
                + "/?noautofilter&operationplan__status__in=approved,confirmed&operationplan__type=PO&quantity__gt=0"
                + "\" onclick='event.stopPropagation()'><span class='ps-2 fa fa-caret-right'></span></a>";
            }
            rows += '</td><td>'
              + grid.formatNumber(thenetwork[5]);

            if (thenetwork[5] != 0) {
              rows += "<a href=\"" + url_prefix + "/data/input/operationplanmaterial/buffer/"
                + admin_escape(thenetwork[0] + " @ " + thenetwork[2])
                + "/?noautofilter&operationplan__status__in=approved,confirmed&operationplan__type=DO"
                + "\" onclick='event.stopPropagation()'><span class='ps-2 fa fa-caret-right'></span></a>";
            }
            rows += '</td><td>'
              + grid.formatNumber(thenetwork[6]);

            if (thenetwork[6] > 0) {
              rows += "<a href=\"" + url_prefix + "/data/input/operationplanmaterial/buffer/"
                + admin_escape(thenetwork[0] + " @ " + thenetwork[2])
                + "/?noautofilter&operationplan__status__in=approved,confirmed&operationplan__type=MO&quantity__gt=0"
                + "\" onclick='event.stopPropagation()'><span class='ps-2 fa fa-caret-right'></span></a>";
            }
            rows += '</td><td>'
              + grid.formatNumber(thenetwork[7]);

            if (thenetwork[7] > 0) {
              rows += "<a href=\"" + url_prefix + "/data/input/demand/?noautofilter&status__in=open,quote&item="
                + admin_escape(thenetwork[0]) + "&location=" + admin_escape(thenetwork[2]) + "&due__lt=" + admin_escape(thenetwork[9])
                + "\" onclick='event.stopPropagation()'><span class='ps-2 fa fa-caret-right'></span></a>";
            }
            rows += '</td><td>'
              + grid.formatNumber(thenetwork[8]);

            if (thenetwork[8] > 0) {
              rows += "<a href=\"" + url_prefix + "/data/input/demand/?noautofilter&status__in=open,quote&item="
                + admin_escape(thenetwork[0]) + "&location=" + admin_escape(thenetwork[2]) + "&due__gte=" + admin_escape(thenetwork[9])
                + "\" onclick='event.stopPropagation()'><span class='ps-2 fa fa-caret-right'></span></a>";
            }
            rows += '</td></tr>';
          });
        }
      }
      angular.element(document).find('#attributes-networkstatus tbody').append(rows);
    }); //watch end

  } //link end
} //directive end
