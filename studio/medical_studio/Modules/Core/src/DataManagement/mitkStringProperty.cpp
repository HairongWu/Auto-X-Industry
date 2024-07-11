/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkStringProperty.h"
#include <nlohmann/json.hpp>

const char *mitk::StringProperty::PATH = "path";
mitk::StringProperty::StringProperty(const char *string) : m_Value()
{
  if (string)
    m_Value = string;
}

mitk::StringProperty::StringProperty(const std::string &s) : m_Value(s)
{
}

mitk::StringProperty::StringProperty(const StringProperty &other) : BaseProperty(other), m_Value(other.m_Value)
{
}

bool mitk::StringProperty::IsEqual(const BaseProperty &property) const
{
  return this->m_Value == static_cast<const Self &>(property).m_Value;
}

bool mitk::StringProperty::Assign(const BaseProperty &property)
{
  this->m_Value = static_cast<const Self &>(property).m_Value;
  return true;
}

std::string mitk::StringProperty::GetValueAsString() const
{
  return m_Value;
}

bool mitk::StringProperty::ToJSON(nlohmann::json& j) const
{
  j = this->GetValueAsString();
  return true;
}

bool mitk::StringProperty::FromJSON(const nlohmann::json& j)
{
  this->SetValue(j.get<std::string>());
  return true;
}

itk::LightObject::Pointer mitk::StringProperty::InternalClone() const
{
  itk::LightObject::Pointer result(new Self(*this));
  result->UnRegister();
  return result;
}
