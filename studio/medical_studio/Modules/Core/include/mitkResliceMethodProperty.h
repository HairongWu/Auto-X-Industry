/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkResliceMethodProperty_h
#define mitkResliceMethodProperty_h

#include "mitkEnumerationProperty.h"

namespace mitk
{
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4522)
#endif

  /**
   * Encapsulates the thick slices method enumeration
   */
  class MITKCORE_EXPORT ResliceMethodProperty : public EnumerationProperty
  {
  public:
    mitkClassMacro(ResliceMethodProperty, EnumerationProperty);
    itkFactorylessNewMacro(Self);
    itkCloneMacro(Self);
    mitkNewMacro1Param(ResliceMethodProperty, const IdType&);
    mitkNewMacro1Param(ResliceMethodProperty, const std::string &);

    using BaseProperty::operator=;

  protected:
    ResliceMethodProperty();
    ResliceMethodProperty(const IdType &value);
    ResliceMethodProperty(const std::string &value);

    void AddThickSlicesTypes();

  private:
    // purposely not implemented
    ResliceMethodProperty &operator=(const ResliceMethodProperty &);

    itk::LightObject::Pointer InternalClone() const override;
  };

#ifdef _MSC_VER
#pragma warning(pop)
#endif

} // end of namespace mitk

#endif
