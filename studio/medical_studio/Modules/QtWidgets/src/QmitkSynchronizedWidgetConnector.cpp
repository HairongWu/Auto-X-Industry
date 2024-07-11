/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

// mitk gui qt common plugin
#include "QmitkSynchronizedWidgetConnector.h"

bool NodeListsEqual(const QmitkSynchronizedWidgetConnector::NodeList& selection1, const QmitkSynchronizedWidgetConnector::NodeList& selection2)
{
  if (selection1.size() != selection2.size())
  {
    return false;
  }

  // lambda to compare node pointer inside both lists
  auto lambda = [](mitk::DataNode::Pointer lhs, mitk::DataNode::Pointer rhs)
  {
    return lhs == rhs;
  };

  return std::is_permutation(selection1.begin(), selection1.end(), selection2.begin(), selection2.end(), lambda);
}

QmitkSynchronizedWidgetConnector::QmitkSynchronizedWidgetConnector()
  : m_SelectAll(true)
  , m_ConnectionCounter(0)
{

}

void QmitkSynchronizedWidgetConnector::ConnectWidget(const QmitkSynchronizedNodeSelectionWidget* nodeSelectionWidget)
{
  connect(nodeSelectionWidget, &QmitkAbstractNodeSelectionWidget::CurrentSelectionChanged,
    this, &QmitkSynchronizedWidgetConnector::ChangeSelection);

  connect(this, &QmitkSynchronizedWidgetConnector::NodeSelectionChanged,
    nodeSelectionWidget, &QmitkAbstractNodeSelectionWidget::SetCurrentSelection);

  connect(nodeSelectionWidget, &QmitkSynchronizedNodeSelectionWidget::SelectionModeChanged,
    this, &QmitkSynchronizedWidgetConnector::ChangeSelectionMode);

  connect(this, &QmitkSynchronizedWidgetConnector::SelectionModeChanged,
    nodeSelectionWidget, &QmitkSynchronizedNodeSelectionWidget::SetSelectAll);

  connect(nodeSelectionWidget, &QmitkSynchronizedNodeSelectionWidget::DeregisterSynchronization,
    this, &QmitkSynchronizedWidgetConnector::DeregisterWidget);

  m_ConnectionCounter++;
}

void QmitkSynchronizedWidgetConnector::DisconnectWidget(const QmitkSynchronizedNodeSelectionWidget* nodeSelectionWidget)
{
  disconnect(nodeSelectionWidget, &QmitkAbstractNodeSelectionWidget::CurrentSelectionChanged,
    this, &QmitkSynchronizedWidgetConnector::ChangeSelection);

  disconnect(this, &QmitkSynchronizedWidgetConnector::NodeSelectionChanged,
    nodeSelectionWidget, &QmitkAbstractNodeSelectionWidget::SetCurrentSelection);

  disconnect(nodeSelectionWidget, &QmitkSynchronizedNodeSelectionWidget::SelectionModeChanged,
    this, &QmitkSynchronizedWidgetConnector::ChangeSelectionMode);

  disconnect(this, &QmitkSynchronizedWidgetConnector::SelectionModeChanged,
    nodeSelectionWidget, &QmitkSynchronizedNodeSelectionWidget::SetSelectAll);

  disconnect(nodeSelectionWidget, &QmitkSynchronizedNodeSelectionWidget::DeregisterSynchronization,
    this, &QmitkSynchronizedWidgetConnector::DeregisterWidget);

  this->DeregisterWidget();
}

void QmitkSynchronizedWidgetConnector::SynchronizeWidget(QmitkSynchronizedNodeSelectionWidget* nodeSelectionWidget) const
{
  // We need to explicitly differentiate when "Select All" is active, since the internal selection
  // might not contain all nodes. When no selection widget is synchronized, the m_InternalSelection
  // won't be updated.
  if (m_SelectAll)
  {
    nodeSelectionWidget->SelectAll();
  }
  else
  {
    nodeSelectionWidget->SetCurrentSelection(m_InternalSelection);
  }

  nodeSelectionWidget->SetSelectAll(m_SelectAll);
}

QmitkSynchronizedWidgetConnector::NodeList QmitkSynchronizedWidgetConnector::GetNodeSelection() const
{
  return m_InternalSelection;
}

bool QmitkSynchronizedWidgetConnector::GetSelectionMode() const
{
  return m_SelectAll;
}

void QmitkSynchronizedWidgetConnector::ChangeSelection(NodeList nodes)
{
  if (!NodeListsEqual(m_InternalSelection, nodes))
  {
    m_InternalSelection = nodes;
    emit NodeSelectionChanged(m_InternalSelection);
  }
}

void QmitkSynchronizedWidgetConnector::ChangeSelectionMode(bool selectAll)
{
  if (m_SelectAll!= selectAll)
  {
    m_SelectAll = selectAll;
    emit SelectionModeChanged(m_SelectAll);
  }
}

void QmitkSynchronizedWidgetConnector::DeregisterWidget()
{
  m_ConnectionCounter--;

  // When no more widgets are synchronized anymore, turn on SelectAll to avoid losing
  // nodes that are added until synchronization of a widget is turned on again.
  if (m_ConnectionCounter == 0)
  {
    m_SelectAll = true;
  }
}
