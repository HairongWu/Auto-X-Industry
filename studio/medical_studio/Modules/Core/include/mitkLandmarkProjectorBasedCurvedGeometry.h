/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkLandmarkProjectorBasedCurvedGeometry_h
#define mitkLandmarkProjectorBasedCurvedGeometry_h

#include "mitkLandmarkProjector.h"

#include "mitkAbstractTransformGeometry.h"
#include "mitkPointSet.h"

namespace mitk
{
  //##Documentation
  //## @brief Superclass of AbstractTransformGeometry sub-classes defined
  //## by a set of landmarks.
  //##
  //## @ingroup Geometry
  class MITKCORE_EXPORT LandmarkProjectorBasedCurvedGeometry : public AbstractTransformGeometry
  {
  public:
    mitkClassMacro(LandmarkProjectorBasedCurvedGeometry, AbstractTransformGeometry);

    void SetLandmarkProjector(mitk::LandmarkProjector *aLandmarkProjector);
    itkGetConstObjectMacro(LandmarkProjector, mitk::LandmarkProjector);

    void SetFrameGeometry(const mitk::BaseGeometry *frameGeometry) override;

    virtual void ComputeGeometry();

    itkGetConstMacro(InterpolatingAbstractTransform, vtkAbstractTransform *);

    itk::LightObject::Pointer InternalClone() const override;

    //##Documentation
    //## @brief Set the landmarks through which the geometry shall pass
    itkSetConstObjectMacro(TargetLandmarks, mitk::PointSet::DataType::PointsContainer);
    //##Documentation
    //## @brief Get the landmarks through which the geometry shall pass
    itkGetConstObjectMacro(TargetLandmarks, mitk::PointSet::DataType::PointsContainer);

  protected:
    LandmarkProjectorBasedCurvedGeometry();

    LandmarkProjectorBasedCurvedGeometry(const LandmarkProjectorBasedCurvedGeometry &other);

    ~LandmarkProjectorBasedCurvedGeometry() override;

    mitk::LandmarkProjector::Pointer m_LandmarkProjector;

    vtkAbstractTransform *m_InterpolatingAbstractTransform;

    mitk::PointSet::DataType::PointsContainer::ConstPointer m_TargetLandmarks;

    //##Documentation
    //## @brief PreSetSpacing
    //##
    //## These virtual function allows a different beahiour in subclasses.
    //## Do implement them in every subclass of BaseGeometry. If not needed, use
    //## {Superclass::PreSetSpacing();};
    void PreSetSpacing(const mitk::Vector3D &aSpacing) override { Superclass::PreSetSpacing(aSpacing); };
  };
} // namespace mitk

#endif
