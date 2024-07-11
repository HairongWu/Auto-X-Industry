/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkRotationOperation_h
#define mitkRotationOperation_h

#include "mitkNumericTypes.h"
#include "mitkOperation.h"

namespace mitk
{
  //##Documentation
  //## @brief Operation, that holds everything necessary for an rotation operation on mitk::BaseData.
  //##
  //## @ingroup Undo
  class MITKCORE_EXPORT RotationOperation : public Operation
  {
  public:
    /**
     * @brief RotationOperation constructor to create the operation.
     * @param operationType this has to be set to OpROTATE.
     * @param pointOfRotation Anchor point for rotation.
     * @param vectorOfRotation Axis for rotation.
     * @param angleOfRotation Angle for rotation in degree.
     */
    RotationOperation(OperationType operationType,
                      Point3D pointOfRotation,
                      Vector3D vectorOfRotation,
                      ScalarType angleOfRotation);
    ~RotationOperation(void) override;

    /**
     * @brief GetAngleOfRotation getter for rotation angle.
     * @return Angle in degree.
     */
    virtual ScalarType GetAngleOfRotation();

    /**
     * @brief GetCenterOfRotation getter for the anchor point of rotation.
     * @return The anchor point to rotate the base data around.
     */
    virtual const Point3D GetCenterOfRotation();

    /**
     * @brief GetVectorOfRotation getter for the rotation axis.
     * @return Rotation axis as vector.
     */
    virtual const Vector3D GetVectorOfRotation();

  protected:
    ScalarType m_AngleOfRotation;
    Point3D m_PointOfRotation;
    Vector3D m_VectorOfRotation;
  };

} // namespace mitk

#endif
