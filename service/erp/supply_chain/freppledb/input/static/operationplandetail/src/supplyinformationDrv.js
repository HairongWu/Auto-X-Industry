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

angular.module('operationplandetailapp').directive('showsupplyinformationDrv', showsupplyinformationDrv);

showsupplyinformationDrv.$inject = ['$window', 'gettextCatalog'];

function showsupplyinformationDrv($window, gettextCatalog) {

  var directive = {
    restrict: 'EA',
    scope: { operationplan: '=data' },
    link: linkfunc
  };
  return directive;

  function linkfunc(scope, elem, attrs) {
    var template = '<div class="card-header"><h5 class="card-title" style="text-transform: capitalize">' +
      gettextCatalog.getString("supply information") +
      '</h5></div>' +
      '<div class="table-responsive"><table class="table table-hover table-sm"><thead><tr><td>' +
      '<b style="text-transform: capitalize;">' + gettextCatalog.getString("priority") + '</b>' +
      '</td><td>' +
      '<b style="text-transform: capitalize;">' + gettextCatalog.getString("types") + '</b>' +
      '</td><td>' +
      '<b style="text-transform: capitalize;">' + gettextCatalog.getString("origin") + '</b>' +
      '</td><td>' +
      '<b style="text-transform: capitalize;">' + gettextCatalog.getString("lead time") + '</b>' +
      '</td><td>' +
      '<b style="text-transform: capitalize;">' + gettextCatalog.getString("cost") + '</b>' +
      '</td><td>' +
      '<b style="text-transform: capitalize;">' + gettextCatalog.getString("size minimum") + '</b>' +
      '</td><td>' +
      '<b style="text-transform: capitalize;">' + gettextCatalog.getString("size multiple") + '</b>' +
      '</td><td>' +
      '<b style="text-transform: capitalize;">' + gettextCatalog.getString("effective start") + '</b>' +
      '</td><td>' +
      '<b style="text-transform: capitalize;">' + gettextCatalog.getString("effective end") + '</b>' +
      '</td></tr></thead>' +
      '<tbody></tbody>' +
      '</table></div>';

    scope.$watchGroup(['operationplan.id', 'operationplan.attributes.supply.length'], function (newValue, oldValue) {
      angular.element(document).find('#attributes-supplyinformation').empty().append(template);
      var rows = '<tr><td colspan="9">' + gettextCatalog.getString('no supply information') + '</td></tr>';

      if (typeof scope.operationplan !== 'undefined') {
        if (scope.operationplan.attributes.hasOwnProperty('supply')) {
          rows = '';
          angular.forEach(scope.operationplan.attributes.supply, function (thesupply) {
            rows += '<tr>'
            for (var i in thesupply) {
              rows += '<td>';

              rows += thesupply[i];

              rows += '</td>';
            }
            rows += '</tr>'
          });
        }
      }
      angular.element(document).find('#attributes-supplyinformation tbody').append(rows);
    }); //watch end

  } //link end
} //directive end
