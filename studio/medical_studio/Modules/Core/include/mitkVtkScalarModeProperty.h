/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkVtkScalarModeProperty_h
#define mitkVtkScalarModeProperty_h

#include "mitkEnumerationProperty.h"

namespace mitk
{
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4522)
#endif

  /**
   * Encapsulates the enumeration vtkInterpolation. Valid values are
   * (VTK constant/Id/string representation):
   * \li VTK_SCALAR_MODE_DEFAULT/0/Default,
   * \li VTK_SCALAR_MODE_USE_POINT_DATA/1/PointData,
   * \li VTK_SCALAR_MODE_USE_CELL_DATA/2/CellData
   * \li VTK_SCALAR_MODE_USE_POINT_FIELD_DATA/3/PointFieldData
   * \li VTK_SCALAR_MODE_USE_CELL_FIELD_DATA/4/CellFieldData
   */
  class MITKCORE_EXPORT VtkScalarModeProperty : public EnumerationProperty
  {
  public:
    mitkClassMacro(VtkScalarModeProperty, EnumerationProperty);

    itkFactorylessNewMacro(Self);

    itkCloneMacro(Self);

      mitkNewMacro1Param(VtkScalarModeProperty, const IdType &);

    mitkNewMacro1Param(VtkScalarModeProperty, const std::string &);

    /**
     * Returns the current scalar mode value as defined by VTK constants.
     * @returns the current scalar mode as VTK constant.
     */
    virtual int GetVtkScalarMode();

    virtual void SetScalarModeToDefault();

    virtual void SetScalarModeToPointData();

    virtual void SetScalarModeToCellData();

    virtual void SetScalarModeToPointFieldData();

    virtual void SetScalarModeToCellFieldData();

    using BaseProperty::operator=;

  protected:
    /**
     * Constructor. Sets the representation to a default value of surface(2)
     */
    VtkScalarModeProperty();

    /**
     * \brief Sets the scalar mode to the given value. If it is not
     * valid, the scalar mode is set to default (0).
     * @param value the integer representation of the scalar mode
     */
    VtkScalarModeProperty(const IdType &value);

    /**
     * \brief Sets the scalar mode to the given value. If it is not
     * valid, the representation is set to default (0).
     * @param value the string representation of the scalar mode
     */
    VtkScalarModeProperty(const std::string &value);

    /**
     * this function is overridden as protected, so that the user may not add
     * additional invalid scalar mode types.
     */
    bool AddEnum(const std::string &name, const IdType &id) override;

    /**
     * Adds the enumeration types as defined by vtk to the list of known
     * enumeration values.
     */
    virtual void AddInterpolationTypes();

  private:
    // purposely not implemented
    VtkScalarModeProperty &operator=(const VtkScalarModeProperty &);

    itk::LightObject::Pointer InternalClone() const override;
  };

#ifdef _MSC_VER
#pragma warning(pop)
#endif

} // end of namespace mitk

#endif
