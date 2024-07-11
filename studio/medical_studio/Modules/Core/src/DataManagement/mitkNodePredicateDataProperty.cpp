/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkNodePredicateDataProperty.h"
#include "mitkDataNode.h"
#include "mitkExceptionMacro.h"

mitk::NodePredicateDataProperty::NodePredicateDataProperty(const char *propertyName, mitk::BaseProperty *p)
: NodePredicateBase(), m_ValidProperty(p), m_ValidPropertyName(propertyName)
{
  if (m_ValidPropertyName.empty())
    mitkThrow() << "NodePredicateDataProperty cannot check invalid (empty) property name";
}

mitk::NodePredicateDataProperty::~NodePredicateDataProperty()
{
}

bool mitk::NodePredicateDataProperty::CheckNode(const mitk::DataNode *node) const
{
  if (node == nullptr)
  {
    mitkThrow() << "NodePredicateDataProperty cannot check invalid node";
  }

  if (m_ValidPropertyName.empty())
  {
    mitkThrow() << "NodePredicateDataProperty cannot check invalid (empty) property name";
  }

  bool result = false;
  auto data = node->GetData();
  if (data)
  {
    if (m_ValidProperty.IsNull())
    {
      result = data->GetProperty(m_ValidPropertyName.c_str()).IsNotNull(); // search only for name
    }
    else
    {
      mitk::BaseProperty::Pointer p = data->GetProperty(m_ValidPropertyName.c_str());

      if (p.IsNotNull())
      {
        result = (*p == *m_ValidProperty);
      }
    }
  }
  return result;
}
