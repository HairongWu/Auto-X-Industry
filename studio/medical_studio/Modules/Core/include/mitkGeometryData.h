/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkGeometryData_h
#define mitkGeometryData_h

#include "mitkBaseData.h"

namespace mitk
{
  //##Documentation
  //## @brief Data class only having a BaseGeometry but not containing
  //## any specific data.
  //##
  //## Only implements pipeline methods which are abstract in BaseData.
  //## @ingroup Geometry
  class MITKCORE_EXPORT GeometryData : public BaseData
  {
  public:
    mitkClassMacro(GeometryData, BaseData);

    itkFactorylessNewMacro(Self);

    itkCloneMacro(Self);

      void UpdateOutputInformation() override;

    void SetRequestedRegionToLargestPossibleRegion() override;

    bool RequestedRegionIsOutsideOfTheBufferedRegion() override;

    bool VerifyRequestedRegion() override;

    void SetRequestedRegion(const itk::DataObject *data) override;

    void CopyInformation(const itk::DataObject *data) override;

  protected:
    GeometryData();

    ~GeometryData() override;
  };

  /**
  * @brief Equal Compare two GeometryData objects for equality, returns true if found equal.
  * @ingroup MITKTestingAPI
  * @param rightHandSide GeometryData to compare.
  * @param leftHandSide GeometryData to compare.
  * @param eps Epsilon to use for floating point comparison. Most of the time mitk::eps will be sufficient.
  * @param verbose Flag indicating if the method should give a detailed console output.
  * @return True if every comparison is true, false in any other case.
  */
  MITKCORE_EXPORT bool Equal(const mitk::GeometryData &leftHandSide,
                             const mitk::GeometryData &rightHandSide,
                             mitk::ScalarType eps,
                             bool verbose);

} // namespace mitk
#endif
