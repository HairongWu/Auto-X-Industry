/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/
#ifndef mitkLookupTableProperty_h
#define mitkLookupTableProperty_h

#include "mitkBaseProperty.h"
#include "mitkLookupTable.h"
#include <MitkCoreExports.h>

namespace mitk
{
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4522)
#endif

  /**
   * @brief The LookupTableProperty class Property to associate mitk::LookupTable
   * to an mitk::DataNode.
   * @ingroup DataManagement
   *
   * @note If you want to use this property to colorize an mitk::Image, make sure
   * to set the mitk::RenderingModeProperty to a mode which supports lookup tables
   * (e.g. LOOKUPTABLE_COLOR). Make sure to check the documentation of the
   * mitk::RenderingModeProperty. For a code example how to use the mitk::LookupTable
   * and this property check the mitkImageVtkMapper2DLookupTableTest.cpp in
   * Core/Code/Testing.
   */
  class MITKCORE_EXPORT LookupTableProperty : public BaseProperty
  {
  protected:
    LookupTable::Pointer m_LookupTable;

    LookupTableProperty();

    LookupTableProperty(const LookupTableProperty &);

    LookupTableProperty(const mitk::LookupTable::Pointer lut);

  public:
    typedef LookupTable::Pointer ValueType;

    mitkClassMacro(LookupTableProperty, BaseProperty);

    itkFactorylessNewMacro(Self);

    itkCloneMacro(Self)
      mitkNewMacro1Param(LookupTableProperty, const mitk::LookupTable::Pointer);

    itkGetObjectMacro(LookupTable, LookupTable);
    ValueType GetValue() const;

    void SetLookupTable(const mitk::LookupTable::Pointer aLookupTable);
    void SetValue(const ValueType &);

    std::string GetValueAsString() const override;

    bool ToJSON(nlohmann::json& j) const override;
    bool FromJSON(const nlohmann::json& j) override;

    using BaseProperty::operator=;

  private:
    // purposely not implemented
    LookupTableProperty &operator=(const LookupTableProperty &);

    itk::LightObject::Pointer InternalClone() const override;

    bool IsEqual(const BaseProperty &property) const override;
    bool Assign(const BaseProperty &property) override;
  };

#ifdef _MSC_VER
#pragma warning(pop)
#endif
} // namespace mitk

#endif
