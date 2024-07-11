/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkPlaneGeometryData_h
#define mitkPlaneGeometryData_h

#include "mitkBaseData.h"
#include "mitkGeometryData.h"
#include "mitkPlaneGeometry.h"
#include <MitkCoreExports.h>

namespace mitk
{
  class PlaneGeometryData;
  /** \deprecatedSince{2014_10} This class is deprecated. Please use PlaneGeometryData instead. */
  DEPRECATED(typedef PlaneGeometryData Geometry2DData);
  //##Documentation
  //## @brief Data class containing PlaneGeometry objects
  //## @ingroup Geometry
  //##
  class MITKCORE_EXPORT PlaneGeometryData : public GeometryData
  {
  public:
    mitkClassMacro(PlaneGeometryData, GeometryData);

    itkFactorylessNewMacro(Self);

    itkCloneMacro(Self);

      //##Documentation
      //## @brief Set the reference to a PlaneGeometry that is stored
      //## by the object
      //##
      //## @warning Accepts only instances of PlaneGeometry or sub-classes.
      void SetGeometry(mitk::BaseGeometry *geometry) override;

    //##Documentation
    //## @brief Set the reference to the PlaneGeometry that is stored
    //## by the object
    virtual void SetPlaneGeometry(mitk::PlaneGeometry *geometry2d);
    /**
    * \deprecatedSince{2014_10} Please use SetPlaneGeometry
    */
    DEPRECATED(void SetGeometry2D(PlaneGeometry *geo)) { SetPlaneGeometry(geo); };
    //##Documentation
    //## @brief Get the reference to the PlaneGeometry that is stored
    //## by the object
    virtual mitk::PlaneGeometry *GetPlaneGeometry() const { return static_cast<mitk::PlaneGeometry *>(GetGeometry()); };
    /**
    * \deprecatedSince{2014_10} Please use GetPlaneGeometry
    */
    DEPRECATED(const PlaneGeometry *GetGeometry2D()) { return GetPlaneGeometry(); };
    void UpdateOutputInformation() override;

    void SetRequestedRegionToLargestPossibleRegion() override;

    bool RequestedRegionIsOutsideOfTheBufferedRegion() override;

    bool VerifyRequestedRegion() override;

    void SetRequestedRegion(const itk::DataObject *data) override;

    void CopyInformation(const itk::DataObject *data) override;

  protected:
    PlaneGeometryData();

    ~PlaneGeometryData() override;
  };
} // namespace mitk
#endif
