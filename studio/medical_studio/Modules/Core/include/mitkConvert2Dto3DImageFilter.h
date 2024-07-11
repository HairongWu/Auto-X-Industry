/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/
#ifndef mitkConvert2Dto3DImageFilter_h
#define mitkConvert2Dto3DImageFilter_h

// MITK
#include "mitkImageToImageFilter.h"
#include <itkImage.h>
#include <mitkImage.h>

namespace mitk
{
  /** \brief Image Filter to convert 2D MITK images to 3D MITK images.
  *
  * A new 3D MITK image is created and all pixel and geometry information from
  * the given 2D input image is copied. The resulting image 3D image has just one slice.
  *
  * This filter can be used when before saving a 2D image with 3D geometry information.
  * By converting it to 3D with just one slice, the common formats (e.g. nrrd) allow
  * a 3x3 transformation matrix.
  *
  * @ingroup Geometry
  */
  class MITKCORE_EXPORT Convert2Dto3DImageFilter : public ImageToImageFilter
  {
  public:
    mitkClassMacro(Convert2Dto3DImageFilter, ImageToImageFilter);
    itkFactorylessNewMacro(Self);
    itkCloneMacro(Self);

      protected :
      /*!
      \brief standard constructor
      */
      Convert2Dto3DImageFilter();
    /*!
    \brief standard destructor
    */
    ~Convert2Dto3DImageFilter() override;
    /*!
    \brief Method generating the output of this filter. Called in the updated process of the pipeline.
    This method generates the smoothed output image.
    */
    void GenerateData() override;

    /*!
    \brief Make a 2D image to a 3D image
    */
    template <typename TPixel, unsigned int VImageDimension>
    void ItkConvert2DTo3D(const itk::Image<TPixel, VImageDimension> *itkImage, mitk::Image::Pointer &mitkImage);
  };
} // END mitk namespace
#endif
