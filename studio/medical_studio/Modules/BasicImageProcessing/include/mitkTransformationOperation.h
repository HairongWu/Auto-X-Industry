/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkTransformationOperation_h
#define mitkTransformationOperation_h

#include <mitkImage.h>
#include <MitkBasicImageProcessingExports.h>
#include <mitkImageMappingHelper.h>

namespace mitk
{

  enum BorderCondition
  {
    Constant,
    Periodic,
    ZeroFluxNeumann
  };

  enum WaveletType
  {
    Held,
    Vow,
    Simoncelli,
    Shannon
  };

  enum GridInterpolationPositionType
  {
    SameSize,
    OriginAligned,
    CenterAligned
  };

  /** \brief Executes a transformation operations on one or two images
  *
  * All parameters of the arithmetic operations must be specified during construction.
  * The actual operation is executed when calling GetResult().
  */
  class MITKBASICIMAGEPROCESSING_EXPORT TransformationOperation {
  public:
    static std::vector<Image::Pointer> MultiResolution(Image::Pointer & image, unsigned int numberOfLevels, bool outputAsDouble = false);
    static Image::Pointer LaplacianOfGaussian(Image::Pointer & image, double sigma, bool outputAsDouble = false);
    static std::vector<Image::Pointer> WaveletForward(Image::Pointer & image, unsigned int numberOfLevels, unsigned int numberOfBands, BorderCondition condition, WaveletType waveletType);

    static Image::Pointer ResampleImage(Image::Pointer &image, mitk::Vector3D spacing, mitk::ImageMappingInterpolator::Type interpolator, GridInterpolationPositionType position, bool returnAsDouble, bool roundOutput);
    static Image::Pointer ResampleMask(Image::Pointer &image, mitk::Vector3D spacing, mitk::ImageMappingInterpolator::Type interpolator, GridInterpolationPositionType position);

  };


}
#endif
