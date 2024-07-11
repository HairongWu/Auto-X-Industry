/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkAnnotationProperty_h
#define mitkAnnotationProperty_h

#include "mitkBaseProperty.h"
#include "mitkNumericTypes.h"
#include <MitkCoreExports.h>

#include <itkConfigure.h>

#include <string>

namespace mitk
{
  /**
   * \brief Property for annotations
   * \ingroup DataManagement
   */
  class MITKCORE_EXPORT AnnotationProperty : public BaseProperty
  {
  public:
    mitkClassMacro(AnnotationProperty, BaseProperty);

    typedef std::string ValueType;

    itkFactorylessNewMacro(Self);

    itkCloneMacro(Self)
      mitkNewMacro2Param(AnnotationProperty, const char *, const Point3D &);
    mitkNewMacro2Param(AnnotationProperty, const std::string &, const Point3D &);
    mitkNewMacro4Param(AnnotationProperty, const char *, ScalarType, ScalarType, ScalarType);
    mitkNewMacro4Param(AnnotationProperty, const std::string &, ScalarType, ScalarType, ScalarType);

    itkGetStringMacro(Label);
    itkSetStringMacro(Label);

    const Point3D &GetPosition() const;
    void SetPosition(const Point3D &position);

    std::string GetValueAsString() const override;

    bool ToJSON(nlohmann::json& j) const override;
    bool FromJSON(const nlohmann::json& j) override;

    virtual BaseProperty &operator=(const BaseProperty &other) { return Superclass::operator=(other); }
    using BaseProperty::operator=;

  protected:
    std::string m_Label;
    Point3D m_Position;

    AnnotationProperty();
    AnnotationProperty(const char *label, const Point3D &position);
    AnnotationProperty(const std::string &label, const Point3D &position);
    AnnotationProperty(const char *label, ScalarType x, ScalarType y, ScalarType z);
    AnnotationProperty(const std::string &label, ScalarType x, ScalarType y, ScalarType z);

    AnnotationProperty(const AnnotationProperty &other);

  private:
    // purposely not implemented
    AnnotationProperty &operator=(const AnnotationProperty &);

    itk::LightObject::Pointer InternalClone() const override;

    bool IsEqual(const BaseProperty &property) const override;
    bool Assign(const BaseProperty &property) override;
  };

} // namespace mitk

#endif
