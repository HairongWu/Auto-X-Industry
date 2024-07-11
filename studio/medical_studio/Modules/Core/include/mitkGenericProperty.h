/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkGenericProperty_h
#define mitkGenericProperty_h

#include <sstream>
#include <cstdlib>
#include <string>

#include "mitkBaseProperty.h"
#include "mitkNumericTypes.h"
#include <MitkCoreExports.h>

namespace mitk
{
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4522)
#endif

  /*!
    @ brief Template class for generating properties for int, float, bool, etc.

    This class template can be instantiated for all classes/internal types that fulfills
    these requirements:
      - an operator<< so that the properties value can be put into a std::stringstream
      - an operator== so that two properties can be checked for equality

    Note: you must use the macros mitkDeclareGenericProperty and mitkDefineGenericProperty to
    provide specializations for concrete types (e.g. BoolProperty). See mitkProperties.h for
    examples. If you don't use these macros, GetNameOfClass() will return "GenericProperty",
    which will mess up serialization for example.

  */
  template <typename T>
  class MITK_EXPORT GenericProperty : public BaseProperty
  {
  public:
    mitkClassMacro(GenericProperty, BaseProperty);
    mitkNewMacro1Param(GenericProperty<T>, T);
    itkCloneMacro(Self);

      typedef T ValueType;

    itkSetMacro(Value, T);
    itkGetConstMacro(Value, T);

    std::string GetValueAsString() const override
    {
      std::stringstream myStr;
      myStr << GetValue();
      return myStr.str();
    }

    bool ToJSON(nlohmann::json&) const override
    {
      return false;
    }

    bool FromJSON(const nlohmann::json&) override
    {
      return false;
    }

    using BaseProperty::operator=;

  protected:
    GenericProperty() {}
    GenericProperty(T x) : m_Value(x) {}
    GenericProperty(const GenericProperty &other) : BaseProperty(other), m_Value(other.m_Value) {}
    T m_Value;

  private:
    // purposely not implemented
    GenericProperty &operator=(const GenericProperty &);

    itk::LightObject::Pointer InternalClone() const override
    {
      itk::LightObject::Pointer result(new Self(*this));
      result->UnRegister();
      return result;
    }

    bool IsEqual(const BaseProperty &other) const override
    {
      return (this->m_Value == static_cast<const Self &>(other).m_Value);
    }

    bool Assign(const BaseProperty &other) override
    {
      this->m_Value = static_cast<const Self &>(other).m_Value;
      return true;
    }
  };

#ifdef _MSC_VER
#pragma warning(pop)
#endif

} // namespace mitk

/**
 * Generates a specialized subclass of mitk::GenericProperty.
 * This way, GetNameOfClass() returns the value provided by PropertyName.
 * Please see mitkProperties.h for examples.
 * @param PropertyName the name of the subclass of GenericProperty
 * @param Type the value type of the GenericProperty
 * @param Export the export macro for DLL usage
 */
#define mitkDeclareGenericProperty(PropertyName, Type, Export)                                                         \
                                                                                                                       \
  class Export PropertyName : public GenericProperty<Type>                                                             \
                                                                                                                       \
  {                                                                                                                    \
  public:                                                                                                              \
    mitkClassMacro(PropertyName, GenericProperty<Type>);                                                               \
    itkFactorylessNewMacro(Self);                                                                                      \
    itkCloneMacro(Self);                                                                                               \
    mitkNewMacro1Param(PropertyName, Type);                                                                            \
                                                                                                                       \
    bool ToJSON(nlohmann::json& j) const override;                                                                     \
    bool FromJSON(const nlohmann::json& j) override;                                                                   \
                                                                                                                       \
    using BaseProperty::operator=;                                                                                     \
                                                                                                                       \
  protected:                                                                                                           \
    PropertyName();                                                                                                    \
    PropertyName(const PropertyName &);                                                                                \
    PropertyName(Type x);                                                                                              \
                                                                                                                       \
  private:                                                                                                             \
    itk::LightObject::Pointer InternalClone() const override;                                                          \
  };

#define mitkDefineGenericProperty(PropertyName, Type, DefaultValue)                                                    \
  mitk::PropertyName::PropertyName() : Superclass(DefaultValue) {}                                                     \
  mitk::PropertyName::PropertyName(const PropertyName &other) : GenericProperty<Type>(other) {}                        \
  mitk::PropertyName::PropertyName(Type x) : Superclass(x) {}                                                          \
  itk::LightObject::Pointer mitk::PropertyName::InternalClone() const                                                  \
  {                                                                                                                    \
    itk::LightObject::Pointer result(new Self(*this));                                                                 \
    result->UnRegister();                                                                                              \
    return result;                                                                                                     \
  }                                                                                                                    \
  bool mitk::PropertyName::ToJSON(nlohmann::json& j) const                                                             \
  {                                                                                                                    \
    j = this->GetValue();                                                                                              \
    return true;                                                                                                       \
  }                                                                                                                    \
                                                                                                                       \
  bool mitk::PropertyName::FromJSON(const nlohmann::json& j)                                                           \
  {                                                                                                                    \
    this->SetValue(j.get<Type>());                                                                                     \
    return true;                                                                                                       \
  }

#endif
