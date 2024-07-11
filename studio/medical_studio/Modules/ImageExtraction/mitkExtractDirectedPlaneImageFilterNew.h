/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkExtractDirectedPlaneImageFilterNew_h
#define mitkExtractDirectedPlaneImageFilterNew_h

#include "itkImage.h"
#include "mitkITKImageImport.h"
#include "mitkImageToImageFilter.h"
#include <MitkImageExtractionExports.h>

namespace mitk
{
  /**
    \deprecated This class is deprecated. Use mitk::ExtractSliceFilter instead.
    \sa ExtractSliceFilter

    \brief A filter that can extract a 2D slice from a 3D or 4D image especially if the image`s axes are rotated

    \sa ContourTool
    \sa SegTool2D
    \sa ExtractImageFilter
    \sa OverwriteDirectedPlaneImageFilter

    \ingroup Process
    \ingroup Reliver

    This class takes an 3D or 4D mitk::Image as input and extracts a slice from it. If you work with a 4D image as input
    you have to specify the
    desired timestep at which the slice shall be extracted, otherwise the lowest given timestep is selected by default.

    The special feature of this filter is, that the planes of the input image can be rotated in any way. To assure a
    proper extraction you have to
    set the currentWorldPlaneGeometry with you can obtain from the BaseRenderer, respectively the positionEvent send by
    the renderer.

    The output will not be set if there was a problem with the input image

    $Author: fetzer $
  */
  class MITKIMAGEEXTRACTION_EXPORT ExtractDirectedPlaneImageFilterNew : public ImageToImageFilter
  {
  public:
    mitkClassMacro(ExtractDirectedPlaneImageFilterNew, ImageToImageFilter);
    itkFactorylessNewMacro(Self);
    itkCloneMacro(Self);

      /**
        \brief Set macro for the current worldgeometry

        \a Parameter The current wordgeometry that describes the position (rotation, translation)
           of the plane (and therefore the slice to be extracted) in our 3D(+t) image
      */
      itkSetMacro(CurrentWorldPlaneGeometry, BaseGeometry *);

    /**
     * \deprecatedSince{2014_10} Please use SetCurrentWorldPlaneGeometry
     */
    DEPRECATED(void SetCurrentWorldGeometry2D(BaseGeometry *geo)) { SetCurrentWorldPlaneGeometry(geo); };
    itkSetMacro(ImageGeometry, BaseGeometry *);

    /**
      \brief Set macro for the current timestep

      \a Parameter The timestep of the image from which the slice shall be extracted
    */
    itkSetMacro(ActualInputTimestep, int);

  protected:
    ExtractDirectedPlaneImageFilterNew();
    ~ExtractDirectedPlaneImageFilterNew() override;
    void GenerateData() override;
    void GenerateOutputInformation() override;

  private:
    const BaseGeometry *m_CurrentWorldPlaneGeometry;
    const BaseGeometry *m_ImageGeometry;
    int m_ActualInputTimestep;

    template <typename TPixel, unsigned int VImageDimension>
    void ItkSliceExtraction(const itk::Image<TPixel, VImageDimension> *inputImage);
  };

} // namespace

#endif
