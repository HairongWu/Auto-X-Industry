/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkGIFImageDescriptionFeatures_h
#define mitkGIFImageDescriptionFeatures_h

#include <mitkAbstractGlobalImageFeature.h>
#include <mitkBaseData.h>
#include <MitkCLUtilitiesExports.h>

namespace mitk
{
  /**
   * \brief Calculates simple features that describe the given image / mask
   *
   * This class can be used to calculated simple image describing features that
   * can be used to compare different images.
   *
   * This feature calculator is activated by the option "<b>-image-diagnostic</b>" or "<b>-id</b>".
   * There are no parameters to further define the behavior of this feature calculator.
   *
   * The parameters for this image are calculated from two images, a mask and an intenstiy
   * image. It is assumed that the mask is
   * of the type of an unsigned short image and all voxels with an value larger or equal
   * to one are treated as masked. (Standard MITK mask)
   *
   * The definition of X, Y, and Z axis depends on the image format and does not necessarily
   * correlates with axial, coronal, and sagittal.
   *
   * The features of this calculator are:
   * - <b>Diagnostic::Dimension X</b>: The number of voxels in the intensity image along the first image dimension.
   * - <b>Diagnostic::Image Dimension Y</b>: The number of voxels in the intensity image along the second image dimension.
   * - <b>Diagnostic::Image Dimension Z</b>: The number of voxels in the intensity image along the third image dimension.
   * - <b>Diagnostic::Image Spacing X</b>: The spacing of the intensity image in the first image dimension.
   * - <b>Diagnostic::Image Spacing Y</b>: The spacing of the intensity image in the second image dimension.
   * - <b>Diagnostic::Image Spacing Z</b>: The spacing of the intensity image in the third image dimension.
   * - <b>Diagnostic::Image Mean intensity</b>: The mean intensity of the whole image.
   * - <b>Diagnostic::Image Minimum intensity</b>: The minimum intensity that occurs in the whole image.
   * - <b>Diagnostic::Image Maximum intensity</b>: The maximum intensity that occurs in the whole image.
   * - <b>Diagnostic::Mask Dimension X</b>: The number of voxels in the mask image along the first image dimension.
   * - <b>Diagnostic::Mask Dimension Y</b>: The number of voxels in the mask image along the second image dimension.
   * - <b>Diagnostic::Mask Dimension Z</b>: The number of voxels in the mask image along the third image dimension.
   * - <b>Diagnostic::Mask bounding box X</b>: The distance between the maximum and minimum position of a masked voxel along the first axis.
   * - <b>Diagnostic::Mask bounding box X</b>: The distance between the maximum and minimum position of a masked voxel along the second axis.
   * - <b>Diagnostic::Mask bounding box X</b>: The distance between the maximum and minimum position of a masked voxel along the third axis.
   * - <b>Diagnostic::Mask Spacing X</b>: The spacing of the mask image in the first image dimension.
   * - <b>Diagnostic::Mask Spacing Y</b>: The spacing of the mask image in the second image dimension.
   * - <b>Diagnostic::Mask Spacing Z</b>: The spacing of the mask image in the third image dimension.
   * - <b>Diagnostic::Mask Voxel count</b>: The number of voxels that are masked.
   * - <b>Diagnostic::Mask Mean intensity</b>: The mean intensity of all voxel that are masked. Also depends on the intensity image.
   * - <b>Diagnostic::Mask Minimum intensity</b> The minimum masked intensity in the intensity image.
   * - <b>Diagnostic::Mask Maximum intensity</b> The maximum masked intensity in the intensity image.
   */
  class MITKCLUTILITIES_EXPORT GIFImageDescriptionFeatures : public AbstractGlobalImageFeature
  {
  public:
    mitkClassMacro(GIFImageDescriptionFeatures,AbstractGlobalImageFeature);
    itkFactorylessNewMacro(Self);
    itkCloneMacro(Self);

    GIFImageDescriptionFeatures();

    FeatureListType CalculateFeatures(const Image* image, const Image* mask, const Image* maskNoNAN) override;
    using Superclass::CalculateFeatures;

    void AddArguments(mitkCommandLineParser& parser) const override;

  protected:

    FeatureListType DoCalculateFeatures(const Image* image, const Image* mask) override;
  };
}
#endif
