/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkStringProperty_h
#define mitkStringProperty_h

#include <itkConfigure.h>

#include "mitkBaseProperty.h"
#include <MitkCoreExports.h>

#include <string>

namespace mitk
{
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4522)
#endif

  /**
   * @brief Property for strings
   * @ingroup DataManagement
   */
  class MITKCORE_EXPORT StringProperty : public BaseProperty
  {
  protected:
    std::string m_Value;

    StringProperty(const char *string = nullptr);
    StringProperty(const std::string &s);

    StringProperty(const StringProperty &);

  public:
    mitkClassMacro(StringProperty, BaseProperty);
    typedef std::string ValueType;

    itkFactorylessNewMacro(Self);
    itkCloneMacro(Self);
    mitkNewMacro1Param(StringProperty, const char*);
    mitkNewMacro1Param(StringProperty, const std::string&);

    itkGetStringMacro(Value);
    itkSetStringMacro(Value);

    std::string GetValueAsString() const override;

    bool ToJSON(nlohmann::json& j) const override;
    bool FromJSON(const nlohmann::json& j) override;

    static const char *PATH;

    using BaseProperty::operator=;

  private:
    // purposely not implemented
    StringProperty &operator=(const StringProperty &);

    itk::LightObject::Pointer InternalClone() const override;

    bool IsEqual(const BaseProperty &property) const override;
    bool Assign(const BaseProperty &property) override;
  };

#ifdef _MSC_VER
#pragma warning(pop)
#endif

} // namespace mitk

#endif
