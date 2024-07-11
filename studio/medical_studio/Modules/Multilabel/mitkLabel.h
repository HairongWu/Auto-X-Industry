/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkLabel_h
#define mitkLabel_h

#include "MitkMultilabelExports.h"
#include <mitkColorProperty.h>
#include <mitkPropertyList.h>
#include <mitkPoint.h>
#include <mitkVector.h>

namespace mitk
{
  //##
  //##Documentation
  //## @brief A data structure describing a label.
  //## @ingroup Data
  //##
  class MITKMULTILABEL_EXPORT Label : public PropertyList
  {
  public:
    mitkClassMacro(Label, mitk::PropertyList);

    typedef unsigned short PixelType;

    itkNewMacro(Self);
    mitkNewMacro2Param(Self, PixelType, const std::string&);

    /// The maximum value a label can get: Since the value is of type unsigned short MAX_LABEL_VALUE = 65535
    static const PixelType MAX_LABEL_VALUE;

    //** Value indicating pixels that are not labeled at all.*/
    static constexpr PixelType UNLABELED_VALUE = 0;

    void SetLocked(bool locked);
    bool GetLocked() const;

    void SetVisible(bool visible);
    bool GetVisible() const;

    void SetOpacity(float opacity);
    float GetOpacity() const;

    void SetName(const std::string &name);
    std::string GetName() const;

    std::string GetTrackingID() const;

    void SetCenterOfMassIndex(const mitk::Point3D &center);
    mitk::Point3D GetCenterOfMassIndex() const;

    void SetCenterOfMassCoordinates(const mitk::Point3D &center);
    mitk::Point3D GetCenterOfMassCoordinates() const;

    void SetColor(const mitk::Color &);
    const mitk::Color &GetColor() const;

    void SetValue(PixelType pixelValue);
    PixelType GetValue() const;

    void SetLayer(unsigned int layer);
    unsigned int GetLayer() const;

    void SetProperty(const std::string &propertyKey, BaseProperty *property, const std::string &contextName = "", bool fallBackOnDefaultContext = false) override;

    using itk::Object::Modified;
    void Modified() { Superclass::Modified(); }
    Label();
    Label(PixelType value, const std::string& name);
    ~Label() override;

  protected:
    void PrintSelf(std::ostream &os, itk::Indent indent) const override;

    Label(const Label &other);

  private:
    itk::LightObject::Pointer InternalClone() const override;
  };

  using LabelVector = std::vector<Label::Pointer>;
  using ConstLabelVector = std::vector<Label::ConstPointer>;

  /**
  * @brief Equal A function comparing two labels for being equal in data
  *
  * @ingroup MITKTestingAPI
  *
  * Following aspects are tested for equality:
  *  - Lebel equality via Equal-PropetyList
  *
  * @param rightHandSide An image to be compared
  * @param leftHandSide An image to be compared
  * @param eps Tolarence for comparison. You can use mitk::eps in most cases.
  * @param verbose Flag indicating if the user wants detailed console output or not.
  * @return true, if all subsequent comparisons are true, false otherwise
  */
  MITKMULTILABEL_EXPORT bool Equal(const mitk::Label &leftHandSide,
                                   const mitk::Label &rightHandSide,
                                   ScalarType eps,
                                   bool verbose);

} // namespace mitk

#endif
