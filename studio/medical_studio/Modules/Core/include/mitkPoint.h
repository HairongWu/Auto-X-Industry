/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkPoint_h
#define mitkPoint_h

#include <itkPoint.h>

#include <mitkArray.h>
#include <mitkEqual.h>
#include <mitkNumericConstants.h>

#include <nlohmann/json.hpp>

namespace mitk
{
  //##Documentation
  //##@brief enumeration of the type a point can be
  enum PointSpecificationType
  {
    PTUNDEFINED = 0,
    PTSTART,
    PTCORNER,
    PTEDGE,
    PTEND
  };

  template <class TCoordRep, unsigned int NPointDimension = 3>
  class Point : public itk::Point<TCoordRep, NPointDimension>
  {
  public:
    /** Default constructor has nothing to do. */
    explicit Point<TCoordRep, NPointDimension>() : itk::Point<TCoordRep, NPointDimension>() {}
    /** Pass-through constructors for the Array base class. */
    template <typename TPointValueType>
    explicit Point(const Point<TPointValueType, NPointDimension> &r) : itk::Point<TCoordRep, NPointDimension>(r)
    {
    }

    template <typename TPointValueType>
    explicit Point(const TPointValueType r[NPointDimension]) : itk::Point<TCoordRep, NPointDimension>(r)
    {
    }

    template <typename TPointValueType>
    explicit Point(const TPointValueType &v) : itk::Point<TCoordRep, NPointDimension>(v)
    {
    }

    Point<TCoordRep, NPointDimension>(const mitk::Point<TCoordRep, NPointDimension> &r)
      : itk::Point<TCoordRep, NPointDimension>(r)
    {
    }
    Point<TCoordRep, NPointDimension>(const TCoordRep r[NPointDimension]) : itk::Point<TCoordRep, NPointDimension>(r) {}
    Point<TCoordRep, NPointDimension>(const TCoordRep &v) : itk::Point<TCoordRep, NPointDimension>(v) {}
    Point<TCoordRep, NPointDimension>(const itk::Point<TCoordRep, NPointDimension> &p)
      : itk::Point<TCoordRep, NPointDimension>(p)
    {
    }

    /**
     * Copies the elements from array array to this.
     * Note that this method will assign doubles to floats without complaining!
     *
     * @param array the array whose values shall be copied. Must overload [] operator.
     */
    template <typename ArrayType>
    void FillPoint(const ArrayType &array)
    {
      itk::FixedArray<TCoordRep, NPointDimension> *thisP =
        dynamic_cast<itk::FixedArray<TCoordRep, NPointDimension> *>(this);
      mitk::FillArray<ArrayType, TCoordRep, NPointDimension>(*thisP, array);
    }

    /**
     * Copies the values stored in this point into the array array.
     *
     * @param array the array which should store the values of this.
     */
    template <typename ArrayType>
    void ToArray(ArrayType array) const
    {
      mitk::ToArray<ArrayType, TCoordRep, NPointDimension>(array, *this);
    }
  };

  template <class TCoordRep, unsigned int NPointDimension>
  void to_json(nlohmann::json& j, const Point<TCoordRep, NPointDimension>& p)
  {
    j = nlohmann::json::array();

    for (size_t i = 0; i < NPointDimension; ++i)
      j.push_back(p[i]);
  }

  template <class TCoordRep, unsigned int NPointDimension>
  void from_json(const nlohmann::json& j, Point<TCoordRep, NPointDimension>& p)
  {
    for (size_t i = 0; i < NPointDimension; ++i)
      j.at(i).get_to(p[i]);
  }

  typedef Point<ScalarType, 2> Point2D;
  typedef Point<ScalarType, 3> Point3D;
  typedef Point<ScalarType, 4> Point4D;

  typedef Point<int, 2> Point2I;
  typedef Point<int, 3> Point3I;
  typedef Point<int, 4> Point4I;

  /**
   * @ingroup MITKTestingAPI
   *
   * @param point1 Point to compare.
   * @param point2 Point to compare.
   * @param eps Tolerance for floating point comparison.
   * @param verbose Flag indicating detailed console output.
   * @return True if points are equal.
   */
  template <typename TCoordRep, unsigned int NPointDimension>
  inline bool Equal(const itk::Point<TCoordRep, NPointDimension> &point1,
                    const itk::Point<TCoordRep, NPointDimension> &point2,
                    TCoordRep eps = mitk::eps,
                    bool verbose = false)
  {
    bool isEqual = true;
    typename itk::Point<TCoordRep, NPointDimension>::VectorType diff = point1 - point2;
    for (unsigned int i = 0; i < NPointDimension; i++)
    {
      if (DifferenceBiggerOrEqualEps(diff[i], eps))
      {
        isEqual = false;
        break;
      }
    }

    ConditionalOutputOfDifference(point1, point2, eps, verbose, isEqual);

    return isEqual;
  }

} // namespace mitk

#endif
