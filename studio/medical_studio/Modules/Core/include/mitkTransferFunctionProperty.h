/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/
#ifndef mitkTransferFunctionProperty_h
#define mitkTransferFunctionProperty_h

#include "mitkBaseProperty.h"
#include "mitkTransferFunction.h"

namespace mitk
{
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4522)
#endif

  /**
   * @brief The TransferFunctionProperty class Property class for the mitk::TransferFunction.
   * @ingroup DataManagement
   *
   * @note If you want to use this property for an mitk::Image, make sure
   * to set the mitk::RenderingModeProperty to a mode which supports transfer
   * functions (e.g. COLORTRANSFERFUNCTION_COLOR). Make sure to check the
   * documentation of the mitk::RenderingModeProperty. For a code example how
   * to use the mitk::TransferFunction check the
   * mitkImageVtkMapper2DTransferFunctionTest.cpp in Core/Code/Testing.
   */
  class MITKCORE_EXPORT TransferFunctionProperty : public BaseProperty
  {
  public:
    typedef mitk::TransferFunction::Pointer ValueType;

    mitkClassMacro(TransferFunctionProperty, BaseProperty);

    itkFactorylessNewMacro(Self);

    itkCloneMacro(Self)
      mitkNewMacro1Param(TransferFunctionProperty, mitk::TransferFunction::Pointer);

    itkSetMacro(Value, mitk::TransferFunction::Pointer);
    itkGetConstMacro(Value, mitk::TransferFunction::Pointer);

    std::string GetValueAsString() const override;

    bool ToJSON(nlohmann::json& j) const override;
    bool FromJSON(const nlohmann::json& j) override;

    using BaseProperty::operator=;

  protected:
    mitk::TransferFunction::Pointer m_Value;

    TransferFunctionProperty();
    TransferFunctionProperty(const TransferFunctionProperty &other);

    TransferFunctionProperty(mitk::TransferFunction::Pointer value);

  private:
    // purposely not implemented
    TransferFunctionProperty &operator=(const TransferFunctionProperty &);

    itk::LightObject::Pointer InternalClone() const override;

    bool IsEqual(const BaseProperty &property) const override;
    bool Assign(const BaseProperty &property) override;
  };

#ifdef _MSC_VER
#pragma warning(pop)
#endif

} // namespace mitk

#endif
