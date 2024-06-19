#
# Copyright (C) 2007-2013 by frePPLe bv
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

from django.urls import re_path
from django.views.generic.base import TemplateView

from freppledb import mode

# Automatically add these URLs when the application is installed
autodiscover = True

if mode == "WSGI":
    from . import views
    from . import serializers

    urlpatterns = [
        # Overridable card for kanban and calendar views
        re_path(
            r"^input/card.html$", TemplateView.as_view(template_name="input/card.html")
        ),
        # Model list reports, which override standard admin screens
        re_path(
            r"^data/input/buffer/$",
            views.BufferList.as_view(),
            name="input_buffer_changelist",
        ),
        re_path(
            r"^data/input/resource/$",
            views.ResourceList.as_view(),
            name="input_resource_changelist",
        ),
        re_path(
            r"^data/input/location/$",
            views.LocationList.as_view(),
            name="input_location_changelist",
        ),
        re_path(
            r"^data/input/customer/$",
            views.CustomerList.as_view(),
            name="input_customer_changelist",
        ),
        re_path(
            r"^data/input/demand/$",
            views.DemandList.as_view(),
            name="input_demand_changelist",
        ),
        re_path(
            r"^data/input/item/$",
            views.ItemList.as_view(),
            name="input_item_changelist",
        ),
        re_path(
            r"^data/input/operationresource/$",
            views.OperationResourceList.as_view(),
            name="input_operationresource_changelist",
        ),
        re_path(
            r"^data/input/operationmaterial/$",
            views.OperationMaterialList.as_view(),
            name="input_operationmaterial_changelist",
        ),
        re_path(
            r"^data/input/calendar/$",
            views.CalendarList.as_view(),
            name="input_calendar_changelist",
        ),
        re_path(
            r"^data/input/calendardetail/(.+)/$",
            views.manufacturing.CalendarDetail.as_view(),
            name="input_calendardetail",
        ),
        re_path(
            r"^data/input/calendarbucket/$",
            views.CalendarBucketList.as_view(),
            name="input_calendarbucket_changelist",
        ),
        re_path(
            r"^data/input/operation/$",
            views.OperationList.as_view(),
            name="input_operation_changelist",
        ),
        re_path(
            r"^data/input/setupmatrix/$",
            views.SetupMatrixList.as_view(),
            name="input_setupmatrix_changelist",
        ),
        re_path(
            r"^data/input/setuprule/$",
            views.SetupRuleList.as_view(),
            name="input_setuprule_changelist",
        ),
        re_path(
            r"^data/input/suboperation/$",
            views.SubOperationList.as_view(),
            name="input_suboperation_changelist",
        ),
        re_path(
            r"^data/input/operationdependency/$",
            views.OperationDependencyList.as_view(),
            name="input_operationdepency_changelist",
        ),
        re_path(
            r"^data/input/manufacturingorder/$",
            views.ManufacturingOrderList.as_view(),
            name="input_manufacturingorder_changelist",
        ),
        re_path(
            r"^data/input/manufacturingorder/location/(.+)/$",
            views.ManufacturingOrderList.as_view(),
            name="input_manufacturingorder_by_location",
        ),
        re_path(
            r"^data/input/manufacturingorder/operation/(.+)/$",
            views.ManufacturingOrderList.as_view(),
            name="input_manufacturingorder_by_operation",
        ),
        re_path(
            r"^data/input/manufacturingorder/item/(.+)/$",
            views.ManufacturingOrderList.as_view(),
            name="input_manufacturingorder_by_item",
        ),
        re_path(
            r"^data/input/manufacturingorder/operationplanmaterial/([^/]+)/([^/]+)/(.+)/$",
            views.ManufacturingOrderList.as_view(),
            name="input_manufacturingorder_by_opm",
        ),
        re_path(
            r"^data/input/manufacturingorder/produced/([^/]+)/([^/]+)/([^/]+)/(.+)/$",
            views.ManufacturingOrderList.as_view(),
            name="input_manufacturingorder_by_produced",
        ),
        re_path(
            r"^data/input/manufacturingorder/consumed/([^/]+)/([^/]+)/([^/]+)/(.+)/$",
            views.ManufacturingOrderList.as_view(),
            name="input_manufacturingorder_by_consumed",
        ),
        re_path(
            r"^data/input/purchaseorder/$",
            views.PurchaseOrderList.as_view(),
            name="input_purchaseorder_changelist",
        ),
        re_path(
            r"^data/input/purchaseorder/item/(.+)/$",
            views.PurchaseOrderList.as_view(),
            name="input_purchaseorder_by_item",
        ),
        re_path(
            r"^data/input/purchaseorder/supplier/(.+)/$",
            views.PurchaseOrderList.as_view(),
            name="input_purchaseorder_by_supplier",
        ),
        re_path(
            r"^data/input/purchaseorder/location/(.+)/$",
            views.PurchaseOrderList.as_view(),
            name="input_purchaseorder_by_location",
        ),
        re_path(
            r"^data/input/purchaseorder/operationplanmaterial/([^/]+)/([^/]+)/(.+)/$",
            views.PurchaseOrderList.as_view(),
            name="input_purchaseorder_by_opm",
        ),
        re_path(
            r"^data/input/purchaseorder/produced/([^/]+)/([^/]+)/([^/]+)/(.+)/$",
            views.PurchaseOrderList.as_view(),
            name="input_purchaseorder_by_produced",
        ),
        re_path(
            r"^data/input/distributionorder/$",
            views.DistributionOrderList.as_view(),
            name="input_distributionorder_changelist",
        ),
        re_path(
            r"^data/input/distributionorder/item/(.+)/$",
            views.DistributionOrderList.as_view(),
            name="input_distributionorder_by_item",
        ),
        re_path(
            r"^data/input/distributionorder/location/(.+)/in/$",
            views.DistributionOrderList.as_view(),
            name="input_distributionorder_in_by_location",
        ),
        re_path(
            r"^data/input/distributionorder/location/(.+)/out/$",
            views.DistributionOrderList.as_view(),
            name="input_distributionorder_out_by_location",
        ),
        re_path(
            r"^data/input/distributionorder/operationplanmaterial/([^/]+)/([^/]+)/(.+)/$",
            views.DistributionOrderList.as_view(),
            name="input_distributionorder_by_opm",
        ),
        re_path(
            r"^data/input/distributionorder/produced/([^/]+)/([^/]+)/([^/]+)/(.+)/$",
            views.DistributionOrderList.as_view(),
            name="input_distributionorder_by_produced",
        ),
        re_path(
            r"^data/input/distributionorder/consumed/([^/]+)/([^/]+)/([^/]+)/(.+)/$",
            views.DistributionOrderList.as_view(),
            name="input_distributionorder_by_consumed",
        ),
        re_path(
            r"^data/input/skill/$",
            views.SkillList.as_view(),
            name="input_skill_changelist",
        ),
        re_path(
            r"^data/input/resourceskill/$",
            views.ResourceSkillList.as_view(),
            name="input_resourceskill_changelist",
        ),
        re_path(
            r"^data/input/supplier/$",
            views.SupplierList.as_view(),
            name="input_supplier_changelist",
        ),
        re_path(
            r"^data/input/itemsupplier/$",
            views.ItemSupplierList.as_view(),
            name="input_itemsupplier_changelist",
        ),
        re_path(
            r"^data/input/itemdistribution/$",
            views.ItemDistributionList.as_view(),
            name="input_itemdistribution_changelist",
        ),
        re_path(
            r"^data/input/deliveryorder/item/(.+)/$",
            views.DeliveryOrderList.as_view(),
            name="input_deliveryorder_by_item",
        ),
        re_path(
            r"^data/input/deliveryorder/consumed/([^/]+)/([^/]+)/([^/]+)/(.+)/$",
            views.DeliveryOrderList.as_view(),
            name="input_deliveryorder_by_consumed",
        ),
        re_path(
            r"^data/input/deliveryorder/$",
            views.DeliveryOrderList.as_view(),
            name="input_deliveryorder_changelist",
        ),
        re_path(
            r"^data/input/operationplanmaterial/item/(.+)/$",
            views.InventoryDetail.as_view(),
            name="input_operationplanmaterial_plandetail_by_item",
        ),
        re_path(
            r"^data/input/operationplanmaterial/buffer/(.+)/$",
            views.InventoryDetail.as_view(),
            name="input_operationplanmaterial_plandetail_by_buffer",
        ),
        re_path(
            r"^data/input/operationplanmaterial/$",
            views.InventoryDetail.as_view(),
            name="input_operationplanmaterial_plan",
        ),
        re_path(
            r"^data/input/operationplanresource/resource/(.+)/$",
            views.ResourceDetail.as_view(),
            name="input_operationplanresource_plandetail",
        ),
        re_path(
            r"^data/input/operationplanresource/$",
            views.ResourceDetail.as_view(),
            name="input_operationplanresource_plan",
        ),
        # Special reports
        re_path(
            r"^data/input/buffer/(.+)/create_or_edit/",
            views.CreateOrEditBuffer,
            name="create_or_edit_buffer",
        ),
        re_path(
            r"^supplypath/item/(.+)/$",
            views.UpstreamItemPath.as_view(),
            name="supplypath_item",
        ),
        re_path(
            r"^whereused/item/(.+)/$",
            views.DownstreamItemPath.as_view(),
            name="whereused_item",
        ),
        re_path(
            r"^supplypath/buffer/(.+)/$",
            views.UpstreamBufferPath.as_view(),
            name="supplypath_buffer",
        ),
        re_path(
            r"^whereused/buffer/(.+)/$",
            views.DownstreamBufferPath.as_view(),
            name="whereused_buffer",
        ),
        re_path(
            r"^supplypath/resource/(.+)/$",
            views.UpstreamResourcePath.as_view(),
            name="supplypath_resource",
        ),
        re_path(
            r"^supplypath/demand/(.+)/$",
            views.UpstreamDemandPath.as_view(),
            name="supplypath_demand",
        ),
        re_path(
            r"^whereused/resource/(.+)/$",
            views.DownstreamResourcePath.as_view(),
            name="whereused_resource",
        ),
        re_path(
            r"^supplypath/operation/(.+)/$",
            views.UpstreamOperationPath.as_view(),
            name="supplypath_operation",
        ),
        re_path(
            r"^whereused/operation/(.+)/$",
            views.DownstreamOperationPath.as_view(),
            name="whereused_operation",
        ),
        re_path(r"^search/$", views.search, name="search"),
        re_path(
            r"^operationplan/$",
            views.OperationPlanDetail.as_view(),
            name="operationplandetail",
        ),
        # REST API framework
        re_path(r"^api/input/buffer/$", serializers.BufferAPI.as_view()),
        re_path(r"^api/input/resource/$", serializers.ResourceAPI.as_view()),
        re_path(r"^api/input/location/$", serializers.LocationAPI.as_view()),
        re_path(r"^api/input/customer/$", serializers.CustomerAPI.as_view()),
        re_path(r"^api/input/demand/$", serializers.DemandAPI.as_view()),
        re_path(r"^api/input/item/$", serializers.ItemAPI.as_view()),
        re_path(
            r"^api/input/operationresource/$",
            serializers.OperationResourceAPI.as_view(),
        ),
        re_path(
            r"^api/input/operationmaterial/$",
            serializers.OperationMaterialAPI.as_view(),
        ),
        re_path(
            r"^api/input/operationplanresource/$",
            serializers.OperationPlanResourceAPI.as_view(),
        ),
        re_path(
            r"^api/input/operationplanmaterial/$",
            serializers.OperationPlanMaterialAPI.as_view(),
        ),
        re_path(r"^api/input/calendar/$", serializers.CalendarAPI.as_view()),
        re_path(
            r"^api/input/calendarbucket/$",
            serializers.CalendarBucketAPI.as_view(),
        ),
        re_path(r"^api/input/operation/$", serializers.OperationAPI.as_view()),
        re_path(
            r"^api/input/setupmatrix/$",
            serializers.SetupMatrixAPI.as_view(),
        ),
        re_path(r"^api/input/setuprule/$", serializers.SetupRuleAPI.as_view()),
        re_path(
            r"^api/input/suboperation/$",
            serializers.SubOperationAPI.as_view(),
        ),
        re_path(
            r"^api/input/operationdependency/$",
            serializers.OperationDependencyAPI.as_view(),
        ),
        re_path(
            r"^api/input/manufacturingorder/$",
            serializers.ManufacturingOrderAPI.as_view(),
        ),
        re_path(
            r"^api/input/purchaseorder/$",
            serializers.PurchaseOrderAPI.as_view(),
        ),
        re_path(
            r"^api/input/distributionorder/$",
            serializers.DistributionOrderAPI.as_view(),
        ),
        re_path(
            r"^api/input/deliveryorder/$",
            serializers.DeliveryOrderAPI.as_view(),
        ),
        re_path(r"^api/input/skill/$", serializers.SkillAPI.as_view()),
        re_path(
            r"^api/input/resourceskill/$",
            serializers.ResourceSkillAPI.as_view(),
        ),
        re_path(r"^api/input/supplier/$", serializers.SupplierAPI.as_view()),
        re_path(
            r"^api/input/itemsupplier/$",
            serializers.ItemSupplierAPI.as_view(),
        ),
        re_path(
            r"^api/input/itemdistribution/$",
            serializers.ItemDistributionAPI.as_view(),
        ),
        re_path(
            r"^api/input/buffer/(?P<pk>(.+))/$",
            serializers.BufferdetailAPI.as_view(),
        ),
        re_path(
            r"^api/input/resource/(?P<pk>(.+))/$",
            serializers.ResourcedetailAPI.as_view(),
        ),
        re_path(
            r"^api/input/location/(?P<pk>(.+))/$",
            serializers.LocationdetailAPI.as_view(),
        ),
        re_path(
            r"^api/input/customer/(?P<pk>(.+))/$",
            serializers.CustomerdetailAPI.as_view(),
        ),
        re_path(
            r"^api/input/demand/(?P<pk>(.+))/$",
            serializers.DemanddetailAPI.as_view(),
        ),
        re_path(
            r"^api/input/item/(?P<pk>(.+))/$",
            serializers.ItemdetailAPI.as_view(),
        ),
        re_path(
            r"^api/input/operationresource/(?P<pk>(.+))/$",
            serializers.OperationResourcedetailAPI.as_view(),
        ),
        re_path(
            r"^api/input/operationmaterial/(?P<pk>(.+))/$",
            serializers.OperationMaterialdetailAPI.as_view(),
        ),
        re_path(
            r"^api/input/operationplanresource/(?P<pk>(.+))/$",
            serializers.OperationPlanResourcedetailAPI.as_view(),
        ),
        re_path(
            r"^api/input/operationplanmaterial/(?P<pk>(.+))/$",
            serializers.OperationPlanMaterialdetailAPI.as_view(),
        ),
        re_path(
            r"^api/input/calendar/(?P<pk>(.+))/$",
            serializers.CalendardetailAPI.as_view(),
        ),
        re_path(
            r"^api/input/calendarbucket/(?P<pk>(.+))/$",
            serializers.CalendarBucketdetailAPI.as_view(),
        ),
        re_path(
            r"^api/input/operation/(?P<pk>(.+))/$",
            serializers.OperationdetailAPI.as_view(),
        ),
        re_path(
            r"^api/input/setupmatrix/(?P<pk>(.+))/$",
            serializers.SetupMatrixdetailAPI.as_view(),
        ),
        re_path(
            r"^api/input/setuprule/(?P<pk>(.+))/$",
            serializers.SetupRuledetailAPI.as_view(),
        ),
        re_path(
            r"^api/input/suboperation/(?P<pk>(.+))/$",
            serializers.SubOperationdetailAPI.as_view(),
        ),
        re_path(
            r"^api/input/operationdependency/(?P<pk>(.+))/$",
            serializers.OperationDependencydetailAPI.as_view(),
        ),
        re_path(
            r"^api/input/manufacturingorder/(?P<pk>(.+))/$",
            serializers.ManufacturingOrderdetailAPI.as_view(),
        ),
        re_path(
            r"^api/input/purchaseorder/(?P<pk>(.+))/$",
            serializers.PurchaseOrderdetailAPI.as_view(),
        ),
        re_path(
            r"^api/input/distributionorder/(?P<pk>(.+))/$",
            serializers.DistributionOrderdetailAPI.as_view(),
        ),
        re_path(
            r"^api/input/deliveryorder/(?P<pk>(.+))/$",
            serializers.DeliveryOrderdetailAPI.as_view(),
        ),
        re_path(
            r"^api/input/skill/(?P<pk>(.+))/$",
            serializers.SkilldetailAPI.as_view(),
        ),
        re_path(
            r"^api/input/resourceskill/(?P<pk>(.+))/$",
            serializers.ResourceSkilldetailAPI.as_view(),
        ),
        re_path(
            r"^api/input/supplier/(?P<pk>(.+))/$",
            serializers.SupplierdetailAPI.as_view(),
        ),
        re_path(
            r"^api/input/itemsupplier/(?P<pk>(.+))/$",
            serializers.ItemSupplierdetailAPI.as_view(),
        ),
        re_path(
            r"^api/input/itemdistribution/(?P<pk>(.+))/$",
            serializers.ItemDistributiondetailAPI.as_view(),
        ),
    ]

else:
    from . import services

    svcpatterns = [
        re_path(r"^operationplan/$", services.OperationplanService.as_asgi()),
    ]
