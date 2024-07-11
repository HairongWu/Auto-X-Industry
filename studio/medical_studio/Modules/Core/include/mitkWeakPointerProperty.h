/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkWeakPointerProperty_h
#define mitkWeakPointerProperty_h

#include "itkWeakPointer.h"
#include "mitkBaseProperty.h"
#include <MitkCoreExports.h>

namespace mitk
{
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4522)
#endif

  //##Documentation
  //## @brief Property containing a smart-pointer
  //##
  //## @ingroup DataManagement
  class MITKCORE_EXPORT WeakPointerProperty : public BaseProperty
  {
  public:
    mitkClassMacro(WeakPointerProperty, BaseProperty);

    itkFactorylessNewMacro(Self);

    itkCloneMacro(Self);
    mitkNewMacro1Param(WeakPointerProperty, itk::Object*);

    ~WeakPointerProperty() override;

    typedef itk::WeakPointer<itk::Object> ValueType;

    ValueType GetWeakPointer() const;
    ValueType GetValue() const;

    void SetWeakPointer(itk::Object *pointer);
    void SetValue(const ValueType &value);

    std::string GetValueAsString() const override;

    bool ToJSON(nlohmann::json& j) const override;
    bool FromJSON(const nlohmann::json& j) override;

    using BaseProperty::operator=;

  protected:
    itk::WeakPointer<itk::Object> m_WeakPointer;

    WeakPointerProperty(const WeakPointerProperty &);

    WeakPointerProperty(itk::Object *pointer = nullptr);

  private:
    // purposely not implemented
    WeakPointerProperty &operator=(const WeakPointerProperty &);

    itk::LightObject::Pointer InternalClone() const override;

    bool IsEqual(const BaseProperty &property) const override;
    bool Assign(const BaseProperty &property) override;
  };

#ifdef _MSC_VER
#pragma warning(pop)
#endif

} // namespace mitk

#endif
