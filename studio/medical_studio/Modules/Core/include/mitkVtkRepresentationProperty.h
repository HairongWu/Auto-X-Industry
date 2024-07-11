/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkVtkRepresentationProperty_h
#define mitkVtkRepresentationProperty_h

#include "mitkEnumerationProperty.h"

namespace mitk
{
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4522)
#endif

  /**
   * Encapsulates the enumeration vtkRepresentation. Valid values are
   * (VTK constant/Id/string representation):
   * VTK_POINTS/0/Points, VTK_WIREFRAME/1/Wireframe, VTK_SURFACE/2/Surface
   * Default is the Surface representation
   */
  class MITKCORE_EXPORT VtkRepresentationProperty : public EnumerationProperty
  {
  public:
    mitkClassMacro(VtkRepresentationProperty, EnumerationProperty);

    itkFactorylessNewMacro(Self);

    itkCloneMacro(Self);

      mitkNewMacro1Param(VtkRepresentationProperty, const IdType &);

    mitkNewMacro1Param(VtkRepresentationProperty, const std::string &);

    /**
     * Returns the current representation value as defined by VTK constants.
     * @returns the current representation as VTK constant.
     */
    virtual int GetVtkRepresentation();

    /**
     * Sets the representation type to VTK_POINTS.
     */
    virtual void SetRepresentationToPoints();

    /**
     * Sets the representation type to VTK_WIREFRAME.
     */
    virtual void SetRepresentationToWireframe();

    /**
     * Sets the representation type to VTK_SURFACE.
     */
    virtual void SetRepresentationToSurface();

    using BaseProperty::operator=;

  protected:
    /**
     * Constructor. Sets the representation to a default value of Surface(2)
     */
    VtkRepresentationProperty();

    /**
     * Constructor. Sets the representation to the given value. If it is not
     * valid, the representation is set to Surface(2)
     * @param value the integer representation of the representation
     */
    VtkRepresentationProperty(const IdType &value);

    /**
     * Constructor. Sets the representation to the given value. If it is not
     * valid, the representation is set to Surface(2)
     * @param value the string representation of the representation
     */
    VtkRepresentationProperty(const std::string &value);

    /**
     * this function is overridden as protected, so that the user may not add
     * additional invalid representation types.
     */
    bool AddEnum(const std::string &name, const IdType &id) override;

    /**
     * Adds the enumeration types as defined by vtk to the list of known
     * enumeration values.
     */
    virtual void AddRepresentationTypes();

  private:
    // purposely not implemented
    VtkRepresentationProperty &operator=(const VtkRepresentationProperty &);

    itk::LightObject::Pointer InternalClone() const override;
  };

#ifdef _MSC_VER
#pragma warning(pop)
#endif

} // end of namespace mitk
#endif
