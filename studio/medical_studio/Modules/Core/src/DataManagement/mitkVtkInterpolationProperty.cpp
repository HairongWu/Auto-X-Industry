/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkVtkInterpolationProperty.h"
#include <vtkProperty.h>

mitk::VtkInterpolationProperty::VtkInterpolationProperty()
{
  AddInterpolationTypes();
  SetValue(static_cast<IdType>(VTK_GOURAUD));
}

mitk::VtkInterpolationProperty::VtkInterpolationProperty(const IdType &value)
{
  AddInterpolationTypes();
  if (IsValidEnumerationValue(value))
  {
    SetValue(value);
  }
  else
  {
    SetValue(static_cast<IdType>(VTK_GOURAUD));
  }
}

mitk::VtkInterpolationProperty::VtkInterpolationProperty(const std::string &value)
{
  AddInterpolationTypes();
  if (IsValidEnumerationValue(value))
  {
    SetValue(value);
  }
  else
  {
    SetValue(static_cast<IdType>(VTK_GOURAUD));
  }
}

int mitk::VtkInterpolationProperty::GetVtkInterpolation()
{
  return static_cast<int>(GetValueAsId());
}

void mitk::VtkInterpolationProperty::SetInterpolationToFlat()
{
  SetValue(static_cast<IdType>(VTK_FLAT));
}

void mitk::VtkInterpolationProperty::SetInterpolationToGouraud()
{
  SetValue(static_cast<IdType>(VTK_GOURAUD));
}

void mitk::VtkInterpolationProperty::SetInterpolationToPhong()
{
  SetValue(static_cast<IdType>(VTK_PHONG));
}

void mitk::VtkInterpolationProperty::AddInterpolationTypes()
{
  AddEnum("Flat", static_cast<IdType>(VTK_FLAT));
  AddEnum("Gouraud", static_cast<IdType>(VTK_GOURAUD));
  AddEnum("Phong", static_cast<IdType>(VTK_PHONG));
}

bool mitk::VtkInterpolationProperty::AddEnum(const std::string &name, const IdType &id)
{
  return Superclass::AddEnum(name, id);
}

itk::LightObject::Pointer mitk::VtkInterpolationProperty::InternalClone() const
{
  itk::LightObject::Pointer result(new Self(*this));
  result->UnRegister();
  return result;
}
