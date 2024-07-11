/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkLabelSetImageSurfaceStampFilter.h"

#include "mitkImageAccessByItk.h"
#include "mitkImageCast.h"

#include <mitkLabelSetImage.h>
#include <mitkLabelSetImage.h>
#include <mitkSurface.h>
#include <mitkSurfaceToImageFilter.h>

mitk::LabelSetImageSurfaceStampFilter::LabelSetImageSurfaceStampFilter() : m_ForceOverwrite(false)
{
  this->SetNumberOfIndexedInputs(1);
  this->SetNumberOfRequiredInputs(1);
}

mitk::LabelSetImageSurfaceStampFilter::~LabelSetImageSurfaceStampFilter()
{
}

void mitk::LabelSetImageSurfaceStampFilter::GenerateData()
{
  // GenerateOutputInformation();
  this->SetNthOutput(0, this->GetInput(0));

  mitk::Image::Pointer inputImage = this->GetInput(0);

  if (m_Surface.IsNull())
  {
    MITK_ERROR << "Input surface is nullptr.";
    return;
  }

  mitk::SurfaceToImageFilter::Pointer surfaceToImageFilter = mitk::SurfaceToImageFilter::New();
  surfaceToImageFilter->MakeOutputBinaryOn();
  surfaceToImageFilter->SetInput(m_Surface);
  surfaceToImageFilter->SetImage(inputImage);
  surfaceToImageFilter->Update();
  mitk::Image::Pointer resultImage = surfaceToImageFilter->GetOutput();

  AccessByItk_1(inputImage, ItkImageProcessing, resultImage);
  inputImage->DisconnectPipeline();
}

template <typename TPixel, unsigned int VImageDimension>
void mitk::LabelSetImageSurfaceStampFilter::ItkImageProcessing(itk::Image<TPixel, VImageDimension> *itkImage,
                                                               mitk::Image::Pointer resultImage)
{
  typedef itk::Image<TPixel, VImageDimension> ImageType;
  const mitk::LabelSetImage* labelSetInputImage = dynamic_cast<LabelSetImage *>(GetInput());
  try
  {
    typename ImageType::Pointer itkResultImage = ImageType::New();
    mitk::CastToItkImage(resultImage, itkResultImage);

    typedef itk::ImageRegionConstIterator<ImageType> SourceIteratorType;
    typedef itk::ImageRegionIterator<ImageType> TargetIteratorType;

    SourceIteratorType sourceIter(itkResultImage, itkResultImage->GetLargestPossibleRegion());
    sourceIter.GoToBegin();

    TargetIteratorType targetIter(itkImage, itkImage->GetLargestPossibleRegion());
    targetIter.GoToBegin();

    int activeLabel = labelSetInputImage->GetActiveLabel()->GetValue();

    while (!sourceIter.IsAtEnd())
    {
      auto sourceValue = static_cast<int>(sourceIter.Get());
      auto targetValue = static_cast<int>(targetIter.Get());
      auto label = labelSetInputImage->GetLabel(targetValue);

      if ((sourceValue != LabelSetImage::UNLABELED_VALUE) &&
          (m_ForceOverwrite ||
           label.IsNull() || !label->GetLocked())) // skip unlabeled source pixels and locked target labels
      {
        targetIter.Set(activeLabel);
      }
      ++sourceIter;
      ++targetIter;
    }
  }
  catch (itk::ExceptionObject &e)
  {
    mitkThrow() << e.GetDescription();
  }
  this->Modified();
}

void mitk::LabelSetImageSurfaceStampFilter::GenerateOutputInformation()
{
  mitk::Image::Pointer inputImage = (mitk::Image *)this->GetInput();
  mitk::Image::Pointer output = this->GetOutput();
  itkDebugMacro(<< "GenerateOutputInformation()");
  if (inputImage.IsNull())
    return;
}
