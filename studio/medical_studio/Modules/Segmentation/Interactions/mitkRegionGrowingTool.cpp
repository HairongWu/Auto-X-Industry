/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkRegionGrowingTool.h"
#include "mitkBaseRenderer.h"
#include "mitkImageToContourModelFilter.h"
#include "mitkRegionGrowingTool.xpm"
#include "mitkRenderingManager.h"
#include "mitkToolManager.h"

// us
#include <usGetModuleContext.h>
#include <usModule.h>
#include <usModuleContext.h>
#include <usModuleResource.h>

// ITK
#include "mitkITKImageImport.h"
#include "mitkImageAccessByItk.h"
#include <itkConnectedComponentImageFilter.h>
#include <itkConnectedThresholdImageFilter.h>
#include <itkNeighborhoodIterator.h>

#include <itkImageDuplicator.h>

#include <limits>

namespace mitk
{
  MITK_TOOL_MACRO(MITKSEGMENTATION_EXPORT, RegionGrowingTool, "Region growing tool");
}

#define ROUND(a) ((a) > 0 ? (int)((a) + 0.5) : -(int)(0.5 - (a)))

mitk::RegionGrowingTool::RegionGrowingTool()
  : FeedbackContourTool("PressMoveRelease"),
    m_SeedValue(0),
    m_ScreenYDifference(0),
    m_ScreenXDifference(0),
    m_MouseDistanceScaleFactor(0.5),
    m_PaintingPixelValue(1),
    m_FillFeedbackContour(true),
    m_ConnectedComponentValue(1)
{
}

mitk::RegionGrowingTool::~RegionGrowingTool()
{
}

void mitk::RegionGrowingTool::ConnectActionsAndFunctions()
{
  CONNECT_FUNCTION("PrimaryButtonPressed", OnMousePressed);
  CONNECT_FUNCTION("Move", OnMouseMoved);
  CONNECT_FUNCTION("Release", OnMouseReleased);
}

const char **mitk::RegionGrowingTool::GetXPM() const
{
  return mitkRegionGrowingTool_xpm;
}

us::ModuleResource mitk::RegionGrowingTool::GetIconResource() const
{
  us::Module *module = us::GetModuleContext()->GetModule();
  us::ModuleResource resource = module->GetResource("RegionGrowing.svg");
  return resource;
}

us::ModuleResource mitk::RegionGrowingTool::GetCursorIconResource() const
{
  us::Module *module = us::GetModuleContext()->GetModule();
  us::ModuleResource resource = module->GetResource("RegionGrowing_Cursor.svg");
  return resource;
}

const char *mitk::RegionGrowingTool::GetName() const
{
  return "Region Growing";
}

void mitk::RegionGrowingTool::Activated()
{
  Superclass::Activated();
}

void mitk::RegionGrowingTool::Deactivated()
{
  Superclass::Deactivated();
}

// Get the average pixel value of square/cube with radius=neighborhood around index
template <typename TPixel, unsigned int imageDimension>
void mitk::RegionGrowingTool::GetNeighborhoodAverage(const itk::Image<TPixel, imageDimension> *itkImage,
                                                     const itk::Index<imageDimension>& index,
                                                     ScalarType *result,
                                                     unsigned int neighborhood)
{
  // maybe assert that image dimension is only 2 or 3?
  auto neighborhoodInt = (int)neighborhood;
  TPixel averageValue(0);
  unsigned int numberOfPixels = (2 * neighborhood + 1) * (2 * neighborhood + 1);
  if (imageDimension == 3)
  {
    numberOfPixels *= (2 * neighborhood + 1);
  }

  MITK_DEBUG << "Getting neighborhood of " << numberOfPixels << " pixels around " << index;

  itk::Index<imageDimension> currentIndex;

  for (int i = (0 - neighborhoodInt); i <= neighborhoodInt; ++i)
  {
    currentIndex[0] = index[0] + i;

    for (int j = (0 - neighborhoodInt); j <= neighborhoodInt; ++j)
    {
      currentIndex[1] = index[1] + j;

      if (imageDimension == 3)
      {
        for (int k = (0 - neighborhoodInt); k <= neighborhoodInt; ++k)
        {
          currentIndex[2] = index[2] + k;

          if (itkImage->GetLargestPossibleRegion().IsInside(currentIndex))
          {
            averageValue += itkImage->GetPixel(currentIndex);
          }
          else
          {
            numberOfPixels -= 1;
          }
        }
      }
      else
      {
        if (itkImage->GetLargestPossibleRegion().IsInside(currentIndex))
        {
          averageValue += itkImage->GetPixel(currentIndex);
        }
        else
        {
          numberOfPixels -= 1;
        }
      }
    }
  }

  *result = (ScalarType)averageValue;
  *result /= numberOfPixels;
}

// Do the region growing (i.e. call an ITK filter that does it)
template <typename TPixel, unsigned int imageDimension>
void mitk::RegionGrowingTool::StartRegionGrowing(const itk::Image<TPixel, imageDimension> *inputImage,
                                                 const itk::Index<imageDimension>& seedIndex,
                                                 const std::array<ScalarType, 2>& thresholds,
                                                 mitk::Image::Pointer &outputImage)
{
  MITK_DEBUG << "Starting region growing at index " << seedIndex << " with lower threshold " << thresholds[0]
             << " and upper threshold " << thresholds[1];

  typedef itk::Image<TPixel, imageDimension> InputImageType;
  typedef itk::Image<DefaultSegmentationDataType, imageDimension> OutputImageType;

  typedef itk::ConnectedThresholdImageFilter<InputImageType, OutputImageType> RegionGrowingFilterType;
  typename RegionGrowingFilterType::Pointer regionGrower = RegionGrowingFilterType::New();

  // perform region growing in desired segmented region
  regionGrower->SetInput(inputImage);
  regionGrower->SetSeed(seedIndex);

  regionGrower->SetLower(thresholds[0]);
  regionGrower->SetUpper(thresholds[1]);

  try
  {
    regionGrower->Update();
  }
  catch (...)
  {
    return; // Should we do something?
  }

  typename OutputImageType::Pointer resultImage = regionGrower->GetOutput();

  // Smooth result: Every pixel is replaced by the majority of the neighborhood
  typedef itk::NeighborhoodIterator<OutputImageType> NeighborhoodIteratorType;
  typedef itk::ImageRegionIterator<OutputImageType> ImageIteratorType;

  typename NeighborhoodIteratorType::RadiusType radius;
  radius.Fill(2); // for now, maybe make this something the user can adjust in the preferences?

  typedef itk::ImageDuplicator< OutputImageType > DuplicatorType;
  typename DuplicatorType::Pointer duplicator = DuplicatorType::New();
  duplicator->SetInputImage(resultImage);
  duplicator->Update();

  typename OutputImageType::Pointer resultDup = duplicator->GetOutput();

  NeighborhoodIteratorType neighborhoodIterator(radius, resultDup, resultDup->GetRequestedRegion());
  ImageIteratorType imageIterator(resultImage, resultImage->GetRequestedRegion());

  for (neighborhoodIterator.GoToBegin(), imageIterator.GoToBegin(); !neighborhoodIterator.IsAtEnd();
       ++neighborhoodIterator, ++imageIterator)
  {
    DefaultSegmentationDataType voteYes(0);
    DefaultSegmentationDataType voteNo(0);

    for (unsigned int i = 0; i < neighborhoodIterator.Size(); ++i)
    {
      if (neighborhoodIterator.GetPixel(i) > 0)
      {
        voteYes += 1;
      }
      else
      {
        voteNo += 1;
      }
    }

    if (voteYes > voteNo)
    {
      imageIterator.Set(1);
    }
    else
    {
      imageIterator.Set(0);
    }
  }

  if (resultImage.IsNull())
  {
    MITK_DEBUG << "Region growing result is empty.";
  }

  // Can potentially have multiple regions, use connected component image filter to label disjunct regions
  typedef itk::ConnectedComponentImageFilter<OutputImageType, OutputImageType> ConnectedComponentImageFilterType;
  typename ConnectedComponentImageFilterType::Pointer connectedComponentFilter =
    ConnectedComponentImageFilterType::New();
  connectedComponentFilter->SetInput(resultImage);
  connectedComponentFilter->Update();
  typename OutputImageType::Pointer resultImageCC = connectedComponentFilter->GetOutput();
  m_ConnectedComponentValue = resultImageCC->GetPixel(seedIndex);

  outputImage = mitk::GrabItkImageMemory(resultImageCC);
}

template <typename TPixel, unsigned int imageDimension>
void mitk::RegionGrowingTool::CalculateInitialThresholds(const itk::Image<TPixel, imageDimension>*)
{
  LevelWindow levelWindow;
  this->GetToolManager()->GetReferenceData(0)->GetLevelWindow(levelWindow);

  m_ThresholdExtrema[0] = static_cast<ScalarType>(std::numeric_limits<TPixel>::lowest());
  m_ThresholdExtrema[1] = static_cast<ScalarType>(std::numeric_limits<TPixel>::max());

  const ScalarType lowerWindowBound = std::max(m_ThresholdExtrema[0], levelWindow.GetLowerWindowBound());
  const ScalarType upperWindowBound = std::min(m_ThresholdExtrema[1], levelWindow.GetUpperWindowBound());

  if (m_SeedValue < lowerWindowBound)
  {
    m_InitialThresholds = { m_ThresholdExtrema[0], lowerWindowBound };
  }
  else if (m_SeedValue > upperWindowBound)
  {
    m_InitialThresholds = { upperWindowBound, m_ThresholdExtrema[1] };
  }
  else
  {
    const ScalarType range = 0.1 * (upperWindowBound - lowerWindowBound); // 10% of the visible window

    m_InitialThresholds[0] = std::min(std::max(lowerWindowBound, m_SeedValue - 0.5 * range), upperWindowBound - range);
    m_InitialThresholds[1] = m_InitialThresholds[0] + range;
  }
}

void mitk::RegionGrowingTool::OnMousePressed(StateMachineAction*, InteractionEvent* interactionEvent)
{
  auto* positionEvent = dynamic_cast<mitk::InteractionPositionEvent*>(interactionEvent);
  if (nullptr == positionEvent)
  {
    return;
  }

  m_LastEventSender = positionEvent->GetSender();
  m_LastEventSlice = m_LastEventSender->GetSlice();
  m_LastScreenPosition = Point2I(positionEvent->GetPointerPositionOnScreen());

  // ReferenceSlice is from the underlying image, WorkingSlice from the active segmentation (can be empty)
  m_ReferenceSlice = FeedbackContourTool::GetAffectedReferenceSlice(positionEvent);
  m_WorkingSlice = FeedbackContourTool::GetAffectedWorkingSlice(positionEvent);

  if (m_WorkingSlice.IsNull())
  {
    // can't do anything without a working slice (i.e. a possibly empty segmentation)
    return;
  }

  // Determine if the user clicked inside or outside of the working slice (i.e. the whole volume)
  mitk::BaseGeometry::Pointer workingSliceGeometry;
  workingSliceGeometry = m_WorkingSlice->GetGeometry();
  workingSliceGeometry->WorldToIndex(positionEvent->GetPositionInWorld(), m_SeedPoint);
  itk::Index<2> indexInWorkingSlice2D;
  indexInWorkingSlice2D[0] = m_SeedPoint[0];
  indexInWorkingSlice2D[1] = m_SeedPoint[1];

  if (!workingSliceGeometry->IsIndexInside(m_SeedPoint))
  {
    MITK_DEBUG << "OnMousePressed: point " << positionEvent->GetPositionInWorld() << " (index coordinates "
               << m_SeedPoint << ") is not inside working slice";
    return;
  }

  mitk::BaseGeometry::Pointer referenceSliceGeometry;
  referenceSliceGeometry = m_ReferenceSlice->GetGeometry();
  itk::Index<3> indexInReferenceSlice;
  itk::Index<2> indexInReferenceSlice2D;
  referenceSliceGeometry->WorldToIndex(positionEvent->GetPositionInWorld(), indexInReferenceSlice);
  indexInReferenceSlice2D[0] = indexInReferenceSlice[0];
  indexInReferenceSlice2D[1] = indexInReferenceSlice[1];

  // Get seed neighborhood
  ScalarType averageValue(0);
  AccessFixedDimensionByItk_3(m_ReferenceSlice, GetNeighborhoodAverage, 2, indexInReferenceSlice2D, &averageValue, 1);
  m_SeedValue = averageValue;
  MITK_DEBUG << "Seed value is " << m_SeedValue;

  // Calculate initial thresholds
  AccessFixedDimensionByItk(m_ReferenceSlice, CalculateInitialThresholds, 2);
  m_Thresholds[0] = m_InitialThresholds[0];
  m_Thresholds[1] = m_InitialThresholds[1];

  // Perform region growing
  mitk::Image::Pointer resultImage = mitk::Image::New();
  AccessFixedDimensionByItk_3(
    m_ReferenceSlice, StartRegionGrowing, 2, indexInWorkingSlice2D, m_Thresholds, resultImage);
  resultImage->SetGeometry(workingSliceGeometry);

  // Extract contour
  if (resultImage.IsNotNull() && m_ConnectedComponentValue >= 1)
  {
    float isoOffset = 0.33;

    mitk::ImageToContourModelFilter::Pointer contourExtractor = mitk::ImageToContourModelFilter::New();
    contourExtractor->SetInput(resultImage);
    contourExtractor->SetContourValue(m_ConnectedComponentValue - isoOffset);
    contourExtractor->Update();
    ContourModel::Pointer resultContour = ContourModel::New();
    resultContour = contourExtractor->GetOutput();

    // Show contour
    if (resultContour.IsNotNull())
    {
      ContourModel::Pointer resultContourWorld = FeedbackContourTool::BackProjectContourFrom2DSlice(
        workingSliceGeometry, FeedbackContourTool::ProjectContourTo2DSlice(m_WorkingSlice, resultContour));

      FeedbackContourTool::UpdateCurrentFeedbackContour(resultContourWorld);

      FeedbackContourTool::SetFeedbackContourVisible(true);
      mitk::RenderingManager::GetInstance()->RequestUpdate(m_LastEventSender->GetRenderWindow());
    }
  }
}

void mitk::RegionGrowingTool::OnMouseMoved(StateMachineAction*, InteractionEvent* interactionEvent)
{
  auto* positionEvent = dynamic_cast<mitk::InteractionPositionEvent*>(interactionEvent);
  if (nullptr == positionEvent)
  {
    return;
  }

  if (m_ReferenceSlice.IsNull())
  {
    return;
  }

  // Get geometry and indices
  mitk::BaseGeometry::Pointer workingSliceGeometry;
  workingSliceGeometry = m_WorkingSlice->GetGeometry();
  itk::Index<2> indexInWorkingSlice2D;
  indexInWorkingSlice2D[0] = m_SeedPoint[0];
  indexInWorkingSlice2D[1] = m_SeedPoint[1];

  m_ScreenYDifference += positionEvent->GetPointerPositionOnScreen()[1] - m_LastScreenPosition[1];
  m_ScreenXDifference += positionEvent->GetPointerPositionOnScreen()[0] - m_LastScreenPosition[0];
  m_LastScreenPosition = Point2I(positionEvent->GetPointerPositionOnScreen());

  // Moving the mouse up and down adjusts the width of the threshold window,
  // moving it left and right shifts the threshold window
  m_Thresholds[0] = std::min(
    m_SeedValue, m_InitialThresholds[0] - (m_ScreenYDifference - m_ScreenXDifference) * m_MouseDistanceScaleFactor);
  m_Thresholds[1] = std::max(
    m_SeedValue, m_InitialThresholds[1] + (m_ScreenYDifference + m_ScreenXDifference) * m_MouseDistanceScaleFactor);

  // Do not exceed the pixel type extrema of the reference slice, though
  m_Thresholds[0] = std::max(m_ThresholdExtrema[0], m_Thresholds[0]);
  m_Thresholds[1] = std::min(m_ThresholdExtrema[1], m_Thresholds[1]);

  // Perform region growing again and show the result
  mitk::Image::Pointer resultImage = mitk::Image::New();
  AccessFixedDimensionByItk_3(
    m_ReferenceSlice, StartRegionGrowing, 2, indexInWorkingSlice2D, m_Thresholds, resultImage);
  resultImage->SetGeometry(workingSliceGeometry);

  // Update the contour
  if (resultImage.IsNotNull() && m_ConnectedComponentValue >= 1)
  {
    float isoOffset = 0.33;

    mitk::ImageToContourModelFilter::Pointer contourExtractor = mitk::ImageToContourModelFilter::New();
    contourExtractor->SetInput(resultImage);
    contourExtractor->SetContourValue(m_ConnectedComponentValue - isoOffset);
    contourExtractor->Update();
    ContourModel::Pointer resultContour = ContourModel::New();
    resultContour = contourExtractor->GetOutput();

    // Show contour
    if (resultContour.IsNotNull())
    {
      ContourModel::Pointer resultContourWorld = FeedbackContourTool::BackProjectContourFrom2DSlice(
        workingSliceGeometry, FeedbackContourTool::ProjectContourTo2DSlice(m_WorkingSlice, resultContour));

      FeedbackContourTool::UpdateCurrentFeedbackContour(resultContourWorld);

      FeedbackContourTool::SetFeedbackContourVisible(true);
      mitk::RenderingManager::GetInstance()->ForceImmediateUpdate(positionEvent->GetSender()->GetRenderWindow());
    }
  }
}

void mitk::RegionGrowingTool::OnMouseReleased(StateMachineAction*, InteractionEvent* interactionEvent)
{
  auto* positionEvent = dynamic_cast<mitk::InteractionPositionEvent*>(interactionEvent);
  if (nullptr == positionEvent)
  {
    return;
  }

  if (m_WorkingSlice.IsNull() && m_FillFeedbackContour)
  {
    return;
  }

  if (m_FillFeedbackContour)
  {
    this->WriteBackFeedbackContourAsSegmentationResult(positionEvent, m_PaintingPixelValue);

    m_ScreenYDifference = 0;
    m_ScreenXDifference = 0;
  }
}
