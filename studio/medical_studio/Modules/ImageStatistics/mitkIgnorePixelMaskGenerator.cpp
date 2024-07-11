/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include <mitkIgnorePixelMaskGenerator.h>
#include <mitkImageTimeSelector.h>
#include <mitkImageAccessByItk.h>
#include <itkImageIterator.h>
#include <itkImageConstIterator.h>
#include <mitkITKImageImport.h>

namespace mitk
{
void IgnorePixelMaskGenerator::SetIgnoredPixelValue(RealType pixelValue)
{
    if (pixelValue != m_IgnoredPixelValue)
    {
        m_IgnoredPixelValue = pixelValue;
        this->Modified();
    }
}

unsigned int IgnorePixelMaskGenerator::GetNumberOfMasks() const
{
  return 1;
}

mitk::Image::ConstPointer IgnorePixelMaskGenerator::DoGetMask(unsigned int)
{
    if (IsUpdateRequired())
    {
        if (m_InputImage.IsNull())
        {
            mitkThrow() << "Image not set!";
        }

        if (m_IgnoredPixelValue == std::numeric_limits<RealType>::min())
        {
          mitkThrow() << "IgnotePixelValue not set!";
        }

        auto timeSliceImage = mitk::SelectImageByTimePoint(m_InputImage, m_TimePoint);

        if (timeSliceImage.IsNull()) mitkThrow() << "Cannot generate mask. Passed time point is not supported by input image. Invalid time point: "<< m_TimePoint;

        // update m_InternalMask
        AccessByItk(timeSliceImage, InternalCalculateMask);
        m_InternalMask->SetGeometry(timeSliceImage->GetGeometry());

        this->Modified();
    }
    m_InternalMaskUpdateTime = m_InternalMask->GetMTime();
    return m_InternalMask;
}

template <typename TPixel, unsigned int VImageDimension>
void IgnorePixelMaskGenerator::InternalCalculateMask(const itk::Image<TPixel, VImageDimension>* image)
{
    typedef itk::Image<TPixel, VImageDimension> ImageType;
    typedef itk::Image<unsigned short, VImageDimension> MaskType;

    typename MaskType::Pointer mask = MaskType::New();
    mask->SetOrigin(image->GetOrigin());
    mask->SetSpacing(image->GetSpacing());
    mask->SetLargestPossibleRegion(image->GetLargestPossibleRegion());
    mask->SetBufferedRegion(image->GetBufferedRegion());
    mask->SetDirection(image->GetDirection());
    mask->SetNumberOfComponentsPerPixel(image->GetNumberOfComponentsPerPixel());
    mask->Allocate();
    mask->FillBuffer(1);

    // iterate over image and mask and set mask=1 if image=m_IgnorePixelValue
    itk::ImageRegionConstIterator<ImageType> imageIterator(image, image->GetLargestPossibleRegion());
    itk::ImageRegionIterator<MaskType> maskIterator(mask, mask->GetLargestPossibleRegion());


    for (imageIterator.GoToBegin(); !imageIterator.IsAtEnd(); ++imageIterator, ++maskIterator)
    {
        if (imageIterator.Value() == static_cast<TPixel>(m_IgnoredPixelValue))
        {
            maskIterator.Set(0);
        }
    }

    m_InternalMask = GrabItkImageMemory(mask);
}

bool IgnorePixelMaskGenerator::IsUpdateRequired() const
{
    unsigned long thisClassTimeStamp = this->GetMTime();
    unsigned long internalMaskTimeStamp = m_InternalMask.IsNull() ? 0 : m_InternalMask->GetMTime();
    unsigned long inputImageTimeStamp = m_InputImage->GetMTime();

    if (thisClassTimeStamp > m_InternalMaskUpdateTime) // inputs have changed
    {
        return true;
    }

    if (m_InternalMaskUpdateTime < inputImageTimeStamp) // mask image has changed outside of this class
    {
        return true;
    }

    if (internalMaskTimeStamp > m_InternalMaskUpdateTime) // internal mask has been changed outside of this class
    {
        return true;
    }

    return false;
}

} // end namespace
