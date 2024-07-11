/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkBinaryThresholdBaseTool_h
#define mitkBinaryThresholdBaseTool_h

#include <MitkSegmentationExports.h>

#include <mitkSegWithPreviewTool.h>

#include <mitkCommon.h>
#include <mitkDataNode.h>

#include <itkBinaryThresholdImageFilter.h>
#include <itkImage.h>

namespace mitk
{
  /**
  \brief Base class for binary threshold tools.

  \ingroup ToolManagerEtAl
  \sa mitk::Tool
  \sa QmitkInteractiveSegmentation
  */
  class MITKSEGMENTATION_EXPORT BinaryThresholdBaseTool : public SegWithPreviewTool
  {
  public:
    Message3<double, double, bool> IntervalBordersChanged;
    Message2<ScalarType, ScalarType> ThresholdingValuesChanged;

    mitkClassMacro(BinaryThresholdBaseTool, SegWithPreviewTool);

    virtual void SetThresholdValues(double lower, double upper);

  protected:
    BinaryThresholdBaseTool(); // purposely hidden
    ~BinaryThresholdBaseTool() override;

    itkSetMacro(LockedUpperThreshold, bool);
    itkGetMacro(LockedUpperThreshold, bool);
    itkBooleanMacro(LockedUpperThreshold);

    itkGetMacro(SensibleMinimumThreshold, ScalarType);
    itkGetMacro(SensibleMaximumThreshold, ScalarType);

    void InitiateToolByInput() override;
    void DoUpdatePreview(const Image* inputAtTimeStep, const Image* oldSegAtTimeStep, LabelSetImage* previewImage, TimeStepType timeStep) override;

    template <typename TPixel, unsigned int VImageDimension>
    void ITKThresholding(const itk::Image<TPixel, VImageDimension>* inputImage,
                         LabelSetImage *segmentation,
                         unsigned int timeStep);

  private:
    ScalarType m_SensibleMinimumThreshold;
    ScalarType m_SensibleMaximumThreshold;
    ScalarType m_LowerThreshold;
    ScalarType m_UpperThreshold;

    /** Indicates if the tool should behave like a single threshold tool (true)
      or like a upper/lower threshold tool (false)*/
    bool m_LockedUpperThreshold = false;

  };

} // namespace

#endif
