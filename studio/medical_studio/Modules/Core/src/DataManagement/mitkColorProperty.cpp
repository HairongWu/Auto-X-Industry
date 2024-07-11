/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkColorProperty.h"
#include <sstream>
#include <nlohmann/json.hpp>

mitk::ColorProperty::ColorProperty() : m_Color(0.0f)
{
}

mitk::ColorProperty::ColorProperty(const mitk::ColorProperty &other) : BaseProperty(other), m_Color(other.m_Color)
{
}

mitk::ColorProperty::ColorProperty(const float color[3]) : m_Color(color)
{
}

mitk::ColorProperty::ColorProperty(const float red, const float green, const float blue)
{
  m_Color.Set(red, green, blue);
}

mitk::ColorProperty::ColorProperty(const mitk::Color &color) : m_Color(color)
{
}

bool mitk::ColorProperty::IsEqual(const BaseProperty &property) const
{
  return this->m_Color == static_cast<const Self &>(property).m_Color;
}

bool mitk::ColorProperty::Assign(const BaseProperty &property)
{
  this->m_Color = static_cast<const Self &>(property).m_Color;
  return true;
}

const mitk::Color &mitk::ColorProperty::GetColor() const
{
  return m_Color;
}

void mitk::ColorProperty::SetColor(const mitk::Color &color)
{
  if (m_Color != color)
  {
    m_Color = color;
    Modified();
  }
}

void mitk::ColorProperty::SetValue(const mitk::Color &color)
{
  SetColor(color);
}

void mitk::ColorProperty::SetColor(float red, float green, float blue)
{
  float tmp[3] = {red, green, blue};
  SetColor(mitk::Color(tmp));
}

std::string mitk::ColorProperty::GetValueAsString() const
{
  std::stringstream myStr;
  myStr.imbue(std::locale::classic());
  myStr << GetValue();
  return myStr.str();
}
const mitk::Color &mitk::ColorProperty::GetValue() const
{
  return GetColor();
}

bool mitk::ColorProperty::ToJSON(nlohmann::json& j) const
{
  j = this->GetColor();
  return true;
}

bool mitk::ColorProperty::FromJSON(const nlohmann::json& j)
{
  this->SetColor(j.get<Color>());
  return true;
}

itk::LightObject::Pointer mitk::ColorProperty::InternalClone() const
{
  itk::LightObject::Pointer result(new Self(*this));
  result->UnRegister();
  return result;
}
