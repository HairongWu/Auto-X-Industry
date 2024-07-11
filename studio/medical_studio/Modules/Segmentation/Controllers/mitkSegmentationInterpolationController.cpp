/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkSegmentationInterpolationController.h"

#include "mitkImageCast.h"
#include "mitkImageReadAccessor.h"
#include "mitkImageTimeSelector.h"
#include <mitkExtractSliceFilter.h>
#include <mitkImageAccessByItk.h>
//#include <mitkPlaneGeometry.h>

#include <itkCommand.h>
#include <itkImage.h>
#include <itkImageSliceConstIteratorWithIndex.h>

#include <thread>

namespace
{
  // itk::Object provides a const version of AddObserver() (which uses const_cast internally)
  // but not a const version of RemoveObserver().
  void RemoveObserverFromConstObject(const itk::Object* constObject, unsigned long observerTag)
  {
    if (nullptr != constObject)
    {
      auto* object = const_cast<itk::Object*>(constObject);
      object->RemoveObserver(observerTag);
    }
  }
}

mitk::SegmentationInterpolationController::InterpolatorMapType
  mitk::SegmentationInterpolationController::s_InterpolatorForImage; // static member initialization

mitk::SegmentationInterpolationController *mitk::SegmentationInterpolationController::InterpolatorForImage(
  const Image *image)
{
  auto iter = s_InterpolatorForImage.find(image);
  if (iter != s_InterpolatorForImage.end())
  {
    return iter->second;
  }
  else
  {
    return nullptr;
  }
}

mitk::SegmentationInterpolationController::SegmentationInterpolationController()
  : m_SegmentationModifiedObserverTag(std::make_pair(0UL, false)),
    m_BlockModified(false),
    m_2DInterpolationActivated(false),
    m_EnableSliceImageCache(false)
{
}

void mitk::SegmentationInterpolationController::Activate2DInterpolation(bool status)
{
  m_2DInterpolationActivated = status;
}

mitk::SegmentationInterpolationController *mitk::SegmentationInterpolationController::GetInstance()
{
  static mitk::SegmentationInterpolationController::Pointer m_Instance;

  if (m_Instance.IsNull())
  {
    m_Instance = SegmentationInterpolationController::New();
  }
  return m_Instance;
}

mitk::SegmentationInterpolationController::~SegmentationInterpolationController()
{
  // remove this from the list of interpolators
  for (auto iter = s_InterpolatorForImage.begin(); iter != s_InterpolatorForImage.end(); ++iter)
  {
    if (iter->second == this)
    {
      s_InterpolatorForImage.erase(iter);
      break;
    }
  }
}

void mitk::SegmentationInterpolationController::OnImageModified(const itk::EventObject &)
{
  if (!m_BlockModified && m_Segmentation.IsNotNull() && m_2DInterpolationActivated)
  {
    SetSegmentationVolume(m_Segmentation);
  }
}

void mitk::SegmentationInterpolationController::BlockModified(bool block)
{
  m_BlockModified = block;
}

void mitk::SegmentationInterpolationController::SetSegmentationVolume(const Image *segmentation)
{
  // clear old information (remove all time steps
  m_SegmentationCountInSlice.clear();

  // delete this from the list of interpolators
  auto iter = s_InterpolatorForImage.find(segmentation);
  if (iter != s_InterpolatorForImage.end())
  {
    s_InterpolatorForImage.erase(iter);
  }

  if (m_SegmentationModifiedObserverTag.second)
  {
    RemoveObserverFromConstObject(m_Segmentation, m_SegmentationModifiedObserverTag.first);
    m_SegmentationModifiedObserverTag.second = false;
  }

  if (nullptr == segmentation || !segmentation->IsInitialized())
  {
    m_Segmentation = nullptr;
    this->InvokeEvent(itk::AbortEvent());
    return;
  }

  if (segmentation->GetDimension() > 4 || segmentation->GetDimension() < 3)
  {
    itkExceptionMacro("SegmentationInterpolationController needs a 3D-segmentation or 3D+t.");
  }

  m_Segmentation = segmentation;

  auto command = itk::ReceptorMemberCommand<SegmentationInterpolationController>::New();
  command->SetCallbackFunction(this, &SegmentationInterpolationController::OnImageModified);
  m_SegmentationModifiedObserverTag.first = segmentation->AddObserver(itk::ModifiedEvent(), command);
  m_SegmentationModifiedObserverTag.second = true;

  m_SegmentationCountInSlice.resize(m_Segmentation->GetTimeSteps());
  for (unsigned int timeStep = 0; timeStep < m_Segmentation->GetTimeSteps(); ++timeStep)
  {
    m_SegmentationCountInSlice[timeStep].resize(3);
    for (unsigned int dim = 0; dim < 3; ++dim)
    {
      m_SegmentationCountInSlice[timeStep][dim].clear();
      m_SegmentationCountInSlice[timeStep][dim].resize(m_Segmentation->GetDimension(dim));
      m_SegmentationCountInSlice[timeStep][dim].assign(m_Segmentation->GetDimension(dim), 0);
    }
  }

  s_InterpolatorForImage.insert(std::make_pair(m_Segmentation, this));

  // for all timesteps
  // scan whole image
  for (unsigned int timeStep = 0; timeStep < m_Segmentation->GetTimeSteps(); ++timeStep)
  {
    ImageTimeSelector::Pointer timeSelector = ImageTimeSelector::New();
    timeSelector->SetInput(m_Segmentation);
    timeSelector->SetTimeNr(timeStep);
    timeSelector->UpdateLargestPossibleRegion();
    Image::Pointer segmentation3D = timeSelector->GetOutput();
    AccessFixedDimensionByItk_2(segmentation3D, ScanWholeVolume, 3, m_Segmentation, timeStep);
  }

  // PrintStatus();

  SetReferenceVolume(m_ReferenceImage);

  Modified();
}

void mitk::SegmentationInterpolationController::SetReferenceVolume(const Image *referenceImage)
{
  m_ReferenceImage = referenceImage;

  if (m_ReferenceImage.IsNull())
    return;                           // no image set - ignore it then
  assert(m_Segmentation.IsNotNull()); // should never happen

  // ensure the reference image has the same dimensionality and extents as the segmentation image
  if (m_ReferenceImage.IsNull() || m_Segmentation.IsNull() ||
      m_ReferenceImage->GetDimension() != m_Segmentation->GetDimension() ||
      m_ReferenceImage->GetPixelType().GetNumberOfComponents() != 1 ||
      m_Segmentation->GetPixelType().GetNumberOfComponents() != 1)
  {
    MITK_WARN << "Segmentation image has different image characteristics than reference image." << std::endl;
    m_ReferenceImage = nullptr;
    return;
  }

  for (unsigned int dim = 0; dim < m_Segmentation->GetDimension(); ++dim)
    if (m_ReferenceImage->GetDimension(dim) != m_Segmentation->GetDimension(dim))
    {
      MITK_WARN << "original patient image does not match segmentation (different extent in dimension " << dim
                << "), ignoring patient image" << std::endl;
      m_ReferenceImage = nullptr;
      return;
    }
}

void mitk::SegmentationInterpolationController::SetChangedVolume(const Image *sliceDiff, unsigned int timeStep)
{
  if (!sliceDiff)
    return;
  if (sliceDiff->GetDimension() != 3)
    return;

  AccessFixedDimensionByItk_1(sliceDiff, ScanChangedVolume, 3, timeStep);

  // PrintStatus();
  Modified();
}

void mitk::SegmentationInterpolationController::SetChangedSlice(const Image *sliceDiff,
                                                                unsigned int sliceDimension,
                                                                unsigned int sliceIndex,
                                                                unsigned int timeStep)
{
  if (!sliceDiff)
    return;
  if (sliceDimension > 2)
    return;
  if (timeStep >= m_SegmentationCountInSlice.size())
    return;
  if (sliceIndex >= m_SegmentationCountInSlice[timeStep][sliceDimension].size())
    return;

  unsigned int dim0(0);
  unsigned int dim1(1);

  // determine the other two dimensions
  switch (sliceDimension)
  {
    default:
    case 2:
      dim0 = 0;
      dim1 = 1;
      break;
    case 1:
      dim0 = 0;
      dim1 = 2;
      break;
    case 0:
      dim0 = 1;
      dim1 = 2;
      break;
  }

  mitk::ImageReadAccessor readAccess(sliceDiff);
  auto *rawSlice = (unsigned char *)readAccess.GetData();
  if (!rawSlice)
    return;

  AccessFixedDimensionByItk_1(
    sliceDiff, ScanChangedSlice, 2, SetChangedSliceOptions(sliceDimension, sliceIndex, dim0, dim1, timeStep, rawSlice));

  Modified();
}

template <typename DATATYPE>
void mitk::SegmentationInterpolationController::ScanChangedSlice(const itk::Image<DATATYPE, 2> *,
                                                                 const SetChangedSliceOptions &options)
{
  auto *pixelData((DATATYPE *)options.pixelData);

  unsigned int timeStep(options.timeStep);

  unsigned int sliceDimension(options.sliceDimension);
  unsigned int sliceIndex(options.sliceIndex);

  if (sliceDimension > 2)
    return;
  if (sliceIndex >= m_SegmentationCountInSlice[timeStep][sliceDimension].size())
    return;

  unsigned int dim0(options.dim0);
  unsigned int dim1(options.dim1);

  int numberOfPixels(0); // number of pixels in this slice that are not 0

  unsigned int dim0max = m_SegmentationCountInSlice[timeStep][dim0].size();
  unsigned int dim1max = m_SegmentationCountInSlice[timeStep][dim1].size();

  // scan the slice from two directions
  // and set the flags for the two dimensions of the slice
  for (unsigned int v = 0; v < dim1max; ++v)
  {
    for (unsigned int u = 0; u < dim0max; ++u)
    {
      DATATYPE value = *(pixelData + u + v * dim0max);

      assert((signed)m_SegmentationCountInSlice[timeStep][dim0][u] + (signed)value >=
             0); // just for debugging. This must always be true, otherwise some counting is going wrong
      assert((signed)m_SegmentationCountInSlice[timeStep][dim1][v] + (signed)value >= 0);

      m_SegmentationCountInSlice[timeStep][dim0][u] =
        static_cast<unsigned int>(m_SegmentationCountInSlice[timeStep][dim0][u] + value);
      m_SegmentationCountInSlice[timeStep][dim1][v] =
        static_cast<unsigned int>(m_SegmentationCountInSlice[timeStep][dim1][v] + value);
      numberOfPixels += static_cast<int>(value);
    }
  }

  // flag for the dimension of the slice itself
  assert((signed)m_SegmentationCountInSlice[timeStep][sliceDimension][sliceIndex] + numberOfPixels >= 0);
  m_SegmentationCountInSlice[timeStep][sliceDimension][sliceIndex] += numberOfPixels;

  // MITK_INFO << "scan t=" << timeStep << " from (0,0) to (" << dim0max << "," << dim1max << ") (" << pixelData << "-"
  // << pixelData+dim0max*dim1max-1 <<  ") in slice " << sliceIndex << " found " << numberOfPixels << " pixels" <<
  // std::endl;
}

template <typename TPixel, unsigned int VImageDimension>
void mitk::SegmentationInterpolationController::ScanChangedVolume(const itk::Image<TPixel, VImageDimension> *diffImage,
                                                                  unsigned int timeStep)
{
  typedef itk::ImageSliceConstIteratorWithIndex<itk::Image<TPixel, VImageDimension>> IteratorType;

  IteratorType iter(diffImage, diffImage->GetLargestPossibleRegion());
  iter.SetFirstDirection(0);
  iter.SetSecondDirection(1);

  int numberOfPixels(0); // number of pixels in this slice that are not 0

  typename IteratorType::IndexType index;
  unsigned int x = 0;
  unsigned int y = 0;
  unsigned int z = 0;

  iter.GoToBegin();
  while (!iter.IsAtEnd())
  {
    while (!iter.IsAtEndOfSlice())
    {
      while (!iter.IsAtEndOfLine())
      {
        index = iter.GetIndex();

        x = index[0];
        y = index[1];
        z = index[2];

        TPixel value = iter.Get();

        assert((signed)m_SegmentationCountInSlice[timeStep][0][x] + (signed)value >=
               0); // just for debugging. This must always be true, otherwise some counting is going wrong
        assert((signed)m_SegmentationCountInSlice[timeStep][1][y] + (signed)value >= 0);

        m_SegmentationCountInSlice[timeStep][0][x] =
          static_cast<unsigned int>(m_SegmentationCountInSlice[timeStep][0][x] + value);
        m_SegmentationCountInSlice[timeStep][1][y] =
          static_cast<unsigned int>(m_SegmentationCountInSlice[timeStep][1][y] + value);

        numberOfPixels += static_cast<int>(value);

        ++iter;
      }
      iter.NextLine();
    }
    assert((signed)m_SegmentationCountInSlice[timeStep][2][z] + numberOfPixels >= 0);
    m_SegmentationCountInSlice[timeStep][2][z] += numberOfPixels;
    numberOfPixels = 0;

    iter.NextSlice();
  }
}

template <typename DATATYPE>
void mitk::SegmentationInterpolationController::ScanWholeVolume(const itk::Image<DATATYPE, 3> *,
                                                                const Image *volume,
                                                                unsigned int timeStep)
{
  if (!volume)
    return;
  if (timeStep >= m_SegmentationCountInSlice.size())
    return;

  ImageReadAccessor readAccess(volume, volume->GetVolumeData(timeStep));

  for (unsigned int slice = 0; slice < volume->GetDimension(2); ++slice)
  {
    const auto *rawVolume =
      static_cast<const DATATYPE *>(readAccess.GetData()); // we again promise not to change anything, we'll just count
    const DATATYPE *rawSlice = rawVolume + (volume->GetDimension(0) * volume->GetDimension(1) * slice);

    ScanChangedSlice<DATATYPE>(nullptr, SetChangedSliceOptions(2, slice, 0, 1, timeStep, rawSlice));
  }
}

void mitk::SegmentationInterpolationController::PrintStatus()
{
  unsigned int timeStep(0); // if needed, put a loop over time steps around everything, but beware, output will be long

  MITK_INFO << "Interpolator status (timestep 0): dimensions " << m_SegmentationCountInSlice[timeStep][0].size() << " "
            << m_SegmentationCountInSlice[timeStep][1].size() << " " << m_SegmentationCountInSlice[timeStep][2].size()
            << std::endl;

  MITK_INFO << "Slice 0: " << m_SegmentationCountInSlice[timeStep][2][0] << std::endl;

  // row "x"
  for (unsigned int index = 0; index < m_SegmentationCountInSlice[timeStep][0].size(); ++index)
  {
    if (m_SegmentationCountInSlice[timeStep][0][index] > 0)
      MITK_INFO << "O";
    else
      MITK_INFO << ".";
  }
  MITK_INFO << std::endl;

  // rows "y" and "z" (diagonal)
  for (unsigned int index = 1; index < m_SegmentationCountInSlice[timeStep][1].size(); ++index)
  {
    if (m_SegmentationCountInSlice[timeStep][1][index] > 0)
      MITK_INFO << "O";
    else
      MITK_INFO << ".";

    if (m_SegmentationCountInSlice[timeStep][2].size() > index) // if we also have a z value here, then print it, too
    {
      for (unsigned int indent = 1; indent < index; ++indent)
        MITK_INFO << " ";

      if (m_SegmentationCountInSlice[timeStep][2][index] > 0)
        MITK_INFO << m_SegmentationCountInSlice[timeStep][2][index]; //"O";
      else
        MITK_INFO << ".";
    }

    MITK_INFO << std::endl;
  }

  // z indices that are larger than the biggest y index
  for (unsigned int index = m_SegmentationCountInSlice[timeStep][1].size();
       index < m_SegmentationCountInSlice[timeStep][2].size();
       ++index)
  {
    for (unsigned int indent = 0; indent < index; ++indent)
      MITK_INFO << " ";

    if (m_SegmentationCountInSlice[timeStep][2][index] > 0)
      MITK_INFO << m_SegmentationCountInSlice[timeStep][2][index]; //"O";
    else
      MITK_INFO << ".";

    MITK_INFO << std::endl;
  }
}

mitk::Image::Pointer mitk::SegmentationInterpolationController::Interpolate(unsigned int sliceDimension,
                                                                            unsigned int sliceIndex,
                                                                            const mitk::PlaneGeometry *currentPlane,
                                                                            unsigned int timeStep,
                                                                            ShapeBasedInterpolationAlgorithm::Pointer algorithm)
{
  if (m_Segmentation.IsNull() || nullptr == currentPlane)
    return nullptr;

  if (timeStep >= m_SegmentationCountInSlice.size())
    return nullptr;

  if (sliceDimension > 2)
    return nullptr;

  if (0 == sliceIndex)
    return nullptr; // First slice, nothing to interpolate

  const unsigned int lastSliceIndex = m_SegmentationCountInSlice[timeStep][sliceDimension].size() - 1;

  if (lastSliceIndex <= sliceIndex)
    return nullptr; // Last slice, nothing to interpolate

  if (m_SegmentationCountInSlice[timeStep][sliceDimension][sliceIndex] > 0)
    return nullptr; // Slice contains segmentation, nothing to interopolate

  unsigned int lowerBound = 0;
  unsigned int upperBound = 0;
  bool bounds = false;

  for (lowerBound = sliceIndex - 1; ; --lowerBound)
  {
    if (m_SegmentationCountInSlice[timeStep][sliceDimension][lowerBound] > 0)
    {
      bounds = true;
      break;
    }

    if (0 == lowerBound)
      break;
  }

  if (!bounds)
    return nullptr;

  bounds = false;

  for (upperBound = sliceIndex + 1; upperBound <= lastSliceIndex; ++upperBound)
  {
    if (m_SegmentationCountInSlice[timeStep][sliceDimension][upperBound] > 0)
    {
      bounds = true;
      break;
    }
  }

  if (!bounds)
    return nullptr;

  // We have found two neighboring slices with segmentations and made sure that the current slice does not contain anything

  mitk::Image::Pointer lowerSlice;
  mitk::Image::Pointer upperSlice;
  mitk::Image::Pointer resultImage;

  try
  {
    // Extract current slice
    resultImage = this->ExtractSlice(currentPlane, sliceIndex, timeStep);

    // Creating PlaneGeometry for lower slice
    auto reslicePlane = currentPlane->Clone();

    // Transforming the current origin so that it matches the lower slice
    auto origin = currentPlane->GetOrigin();
    m_Segmentation->GetSlicedGeometry(timeStep)->WorldToIndex(origin, origin);
    origin[sliceDimension] = lowerBound;
    m_Segmentation->GetSlicedGeometry(timeStep)->IndexToWorld(origin, origin);
    reslicePlane->SetOrigin(origin);

    // Extract lower slice
    lowerSlice = this->ExtractSlice(reslicePlane, lowerBound, timeStep, true);

    if (lowerSlice.IsNull())
      return nullptr;

    // Transforming the current origin so that it matches the upper slice
    m_Segmentation->GetSlicedGeometry(timeStep)->WorldToIndex(origin, origin);
    origin[sliceDimension] = upperBound;
    m_Segmentation->GetSlicedGeometry(timeStep)->IndexToWorld(origin, origin);
    reslicePlane->SetOrigin(origin);

    // Extract the upper slice
    upperSlice = this->ExtractSlice(reslicePlane, upperBound, timeStep, true);

    if (upperSlice.IsNull())
      return nullptr;
  }
  catch (const std::exception &e)
  {
    MITK_ERROR << "Error in 2D interpolation: " << e.what();
    return nullptr;
  }

  // Interpolation algorithm inputs:
  //   - Two segmentations (guaranteed to be of the same data type)
  //   - Orientation of the segmentations (sliceDimension)
  //   - Position of the two slices (sliceIndices)
  //   - Reference image
  //
  // The interpolation algorithm can use e.g. itk::ImageSliceConstIteratorWithIndex to
  // inspect the reference image at appropriate positions.

  if (algorithm.IsNull())
    algorithm = mitk::ShapeBasedInterpolationAlgorithm::New();

  return algorithm->Interpolate(
    lowerSlice.GetPointer(),
    lowerBound,
    upperSlice.GetPointer(),
    upperBound,
    sliceIndex,
    sliceDimension,
    resultImage,
    timeStep,
    m_ReferenceImage);
}

mitk::Image::Pointer mitk::SegmentationInterpolationController::ExtractSlice(const PlaneGeometry* planeGeometry, unsigned int sliceIndex, unsigned int timeStep, bool cache)
{
  static const auto MAX_CACHE_SIZE = 2 * std::thread::hardware_concurrency();
  const auto key = std::make_pair(sliceIndex, timeStep);

  if (cache && m_EnableSliceImageCache)
  {
    std::lock_guard<std::mutex> lock(m_SliceImageCacheMutex);

    if (0 != m_SliceImageCache.count(key))
      return m_SliceImageCache[key];

    if (MAX_CACHE_SIZE < m_SliceImageCache.size())
      m_SliceImageCache.clear();
  }

  auto extractor = ExtractSliceFilter::New();
  extractor->SetInput(m_Segmentation);
  extractor->SetTimeStep(timeStep);
  extractor->SetResliceTransformByGeometry(m_Segmentation->GetTimeGeometry()->GetGeometryForTimeStep(timeStep));
  extractor->SetVtkOutputRequest(false);
  extractor->SetWorldGeometry(planeGeometry);
  extractor->Update();

  if (cache && m_EnableSliceImageCache)
  {
    std::lock_guard<std::mutex> lock(m_SliceImageCacheMutex);
    m_SliceImageCache[key] = extractor->GetOutput();
  }

  return extractor->GetOutput();
}

void mitk::SegmentationInterpolationController::EnableSliceImageCache()
{
  m_EnableSliceImageCache = true;
}

void mitk::SegmentationInterpolationController::DisableSliceImageCache()
{
  m_EnableSliceImageCache = false;
  m_SliceImageCache.clear();
}
